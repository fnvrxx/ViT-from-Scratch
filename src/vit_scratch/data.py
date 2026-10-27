from pathlib import Path

import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms

CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


def load_datasets(data_dir: str | Path = "data", seed: int = 42):
    """Return disjoint train, validation, and official test datasets."""
    transform = transforms.ToTensor()
    full_train = datasets.FashionMNIST(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    test = datasets.FashionMNIST(
        root=str(data_dir), train=False, download=True, transform=transform
    )
    train, validation = random_split(
        full_train, [54_000, 6_000], generator=torch.Generator().manual_seed(seed)
    )
    return train, validation, test
