import torch.nn as nn


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


if __name__ == "__main__":
    print(AlexNet())
