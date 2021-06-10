import torch
import torch.nn as nn
import numpy as np


class BasicNet(nn.Module):
    """Template superclass for our models."""

    def __init__(self, args):
        super().__init__()
        self.args = args


class AlexNet(BasicNet):
    def __init__(self, args):
        super().__init__(args)
        self.convolutions = nn.Sequential(
            nn.Conv2d(3, 96, (11, 11), stride=(4, 4)),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=(2, 2)),
            nn.BatchNorm2d(96),

            nn.Conv2d(96, 256, (5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=(2, 2)),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 384, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(384, 384, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(384, 256, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=(2, 2)),
            nn.BatchNorm2d(256)
        )
        self.flatten = nn.Flatten()
        self.linears = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.Tanh(),
            nn.Dropout(self.args.dropout),
            nn.Linear(4096, 4096),
            nn.Dropout(self.args.dropout),

            nn.Linear(4096, 2),
            nn.Softmax(dim=1)
        )

        self.to(args.device)

    def forward(self, x):
        x = self.convolutions(x)
        x = self.flatten(x)
        x = self.linears(x)
        return x


class ConvBatchNormAct(nn.Module):
    """Sequential application of convolution, batchnorm, and activations layers, with the last two being optional."""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=0, groups=1, bias=False,
                 bn=True, act=True):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding,
                              groups=groups, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.act(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-excitation block"""

    def __init__(self, in_channels, r=24):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(nn.Conv2d(in_channels, in_channels // r, 1),
                                        nn.SiLU(),
                                        nn.Conv2d(in_channels // r, in_channels, 1),
                                        nn.Sigmoid())

    def forward(self, x):
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y


class DropSample(nn.Module):
    """Drops each sample in x with probability p during training"""

    def __init__(self, p=0):
        super().__init__()

        self.p = p

    def forward(self, x):
        if (not self.p) or (not self.training):
            return x

        batch_size = len(x)
        random_tensor = torch.cuda.FloatTensor(batch_size, 1, 1, 1).uniform_()
        bit_mask = self.p < random_tensor

        x = x.div(1 - self.p)
        x = x * bit_mask
        return x


class MBConvN(nn.Module):
    """MBConv with an expansion factor of N, plus squeeze-and-excitation"""

    def __init__(self, in_channels, out_channels, expansion_factor,
                 kernel_size=3, stride=1, r=24, p=0):
        super().__init__()

        padding = (kernel_size - 1) // 2
        expanded = expansion_factor * in_channels
        self.skip_connection = (in_channels == out_channels) and (stride == 1)

        self.expand_pw = nn.Identity() if (expansion_factor == 1) else ConvBatchNormAct(in_channels, expanded,
                                                                                        kernel_size=1)
        self.depthwise = ConvBatchNormAct(expanded, expanded, kernel_size=kernel_size,
                                          stride=stride, padding=padding, groups=expanded)
        self.se = SEBlock(expanded, r=r)
        self.reduce_pw = ConvBatchNormAct(expanded, out_channels, kernel_size=1,
                                          act=False)
        self.dropsample = DropSample(p)

    def forward(self, x):
        residual = x

        x = self.expand_pw(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.reduce_pw(x)

        if self.skip_connection:
            x = self.dropsample(x)
            x = x + residual

        return x


class MBConv1(MBConvN):
    def __init__(self, n_in, n_out, kernel_size=3,
                 stride=1, r=24, p=0):
        super().__init__(n_in, n_out, expansion_factor=1,
                         kernel_size=kernel_size, stride=stride,
                         r=r, p=p)


class MBConv6(MBConvN):
    def __init__(self, n_in, n_out, kernel_size=3,
                 stride=1, r=24, p=0):
        super().__init__(n_in, n_out, expansion_factor=6,
                         kernel_size=kernel_size, stride=stride,
                         r=r, p=p)


def create_stage(n_in, n_out, num_layers, layer_type,
                 kernel_size=3, stride=1, r=24, p=0):
    """Creates a Sequential consisting of [num_layers] layer_type"""
    layers = [layer_type(n_in, n_out, kernel_size=kernel_size,
                         stride=stride, r=r, p=p)]
    layers += [layer_type(n_out, n_out, kernel_size=kernel_size,
                          r=r, p=p) for _ in range(num_layers - 1)]
    layers = nn.Sequential(*layers)
    return layers


def scale_width(w: int, scale_factor):
    scaled = w * scale_factor

    # Modify the scaled width to be at least 8 and divisible by 8.
    new_w = max(8, int(scaled + 4) - (int(scaled + 4) % 8))

    # Do this thing that the EfficientNet paper said we have to ¯\_(ツ)_/¯
    if new_w < scaled * 0.9:
        return new_w + 8
    else:
        return new_w


class EfficientNet(BasicNet):
    def __init__(self, args, w_factor=1, d_factor=1, out_size=1000):
        super().__init__(args)

        base_widths = [(32, 16), (16, 24), (24, 40),
                       (40, 80), (80, 112), (112, 192),
                       (192, 320), (320, 1280)]
        base_depths = [1, 2, 2, 3, 3, 4, 1]

        scaled_widths = [(scale_width(w[0], w_factor), scale_width(w[1], w_factor))
                         for w in base_widths]
        scaled_depths = [np.ceil(d_factor * d) for d in base_depths]

        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
        strides = [1, 2, 2, 2, 1, 2, 1]
        ps = [0, 0.029, 0.057, 0.086, 0.114, 0.143, 0.171]

        self.stem = ConvBatchNormAct(3, scaled_widths[0][0], stride=2, padding=1)

        stages = []
        for i in range(7):
            layer_type = MBConv1 if (i == 0) else MBConv6
            r = 4 if (i == 0) else 24
            stage = create_stage(*scaled_widths[i], scaled_depths[i],
                                 layer_type, kernel_size=kernel_sizes[i],
                                 stride=strides[i], r=r, p=ps[i])
            stages.append(stage)
        self.stages = nn.Sequential(*stages)

        self.pre_head = ConvBatchNormAct(*scaled_widths[-1], kernel_size=1)

        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Flatten(),
                                  nn.Linear(scaled_widths[-1][1], out_size))

    def feature_extractor(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pre_head(x)
        return x

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.head(x)
        return x


class EfficientNetB0(EfficientNet):
    def __init__(self, out_size=1000):
        w_factor = 1
        d_factor = 1
        super().__init__(w_factor, d_factor, out_size)
