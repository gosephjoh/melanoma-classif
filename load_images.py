"""
CIS 472 Machine Learning - Final Project
load_images.py
Contributors: Joseph Goh
Description:
    Contains the Dataset and DataLoader generation code.
Updated: 06/09/2021
"""


from torch import FloatTensor
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision.io import read_image
import os
import numpy
import pandas as pd


class MelanomaImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_labels["image_name"] = self.img_labels["image_name"] + ".jpg"
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path).type(FloatTensor)
        label = self.img_labels.iloc[idx, 7]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def load_images(data_root: str, batch_size: int, split: float, portion: float, seed=None) -> (DataLoader, DataLoader):
    """
    Specify portion to decide how much of the train set to load.
    Specify split to decide what part of the loaded set is to be used for training and what part is to be used for testing.
    Specify seed for reproducible pseudo-randomness.
    """
    transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    # Load all images into Dataset object.
    complete_set = MelanomaImageDataset(f'{data_root}/train.csv', f'{data_root}/train', transform=transform)
    set_size = len(complete_set)

    # Sample a training set and a testing set using pseudo-randomly generated indices
    indices = list(range(len(complete_set)))
    if seed is not None:
        numpy.random.seed(seed)
    numpy.random.shuffle(indices)
    split_idx = int(numpy.floor(split * set_size * portion))
    stop_idx = int(numpy.floor(set_size * portion))
    train_sampler = SubsetRandomSampler(indices[split_idx:stop_idx])
    test_sampler = SubsetRandomSampler(indices[:split_idx])

    train_loader = DataLoader(complete_set, batch_size=batch_size, shuffle=False, num_workers=4, sampler=train_sampler)
    test_loader = DataLoader(complete_set, batch_size=batch_size, shuffle=False, num_workers=4, sampler=test_sampler)

    print("DataLoader objects created!")
    return train_loader, test_loader
