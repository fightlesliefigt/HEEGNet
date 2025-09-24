## From https://github.com/psmmettes/hpn/blob/master/helper.py and https://github.com/MinaGhadimiAtigh/Hyperbolic-Busemann-Learning/blob/master/helper/helper.py

import os
import torch

import numpy as np
import torch.optim as optim
import torch.utils.data as data

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def load_cifar100(basedir, batch_size, kwargs):
    # Input channels normalization.
    mrgb = [0.507, 0.487, 0.441]
    srgb = [0.267, 0.256, 0.276]
    normalize = transforms.Normalize(mean=mrgb, std=srgb)

    # Load train data.
    trainloader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=basedir + 'cifar100/', train=True,
                          transform=transforms.Compose([
                              transforms.RandomCrop(32, 4),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              normalize,
                          ]), download=True),
        batch_size=batch_size, shuffle=True, **kwargs)

    # Labels to torch.
    trainloader.dataset.train_labels = torch.from_numpy(np.array(trainloader.dataset.targets))

    # Load test data.
    testloader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=basedir + 'cifar100/', train=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              normalize,
                          ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    # Labels to torch.
    testloader.dataset.test_labels = torch.from_numpy(np.array(testloader.dataset.targets))

    return trainloader, testloader


def load_cifar10(basedir, batch_size, kwargs):
    # Input channels normalization.
    mrgb = [0.507, 0.487, 0.441]
    srgb = [0.267, 0.256, 0.276]
    normalize = transforms.Normalize(mean=mrgb, std=srgb)

    # Load train data.
    trainloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=basedir + 'cifar10/', train=True,
                         transform=transforms.Compose([
                             transforms.RandomCrop(32, 4),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             normalize,
                         ]), download=True),
        batch_size=batch_size, shuffle=True, **kwargs)

    # Labels to torch.
    trainloader.dataset.train_labels = torch.from_numpy(np.array(trainloader.dataset.targets))

    # Load test data.
    testloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=basedir + 'cifar10/', train=False,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             normalize,
                         ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    # Labels to torch.
    testloader.dataset.test_labels = torch.from_numpy(np.array(testloader.dataset.targets))

    return trainloader, testloader

