"""
Data loading, augmentation, and DataLoader construction for WBC classification.
"""

from pathlib import Path
from typing import Union

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

# ImageNet statistics (used because we start from ImageNet pretrained weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 240
CLASSES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]


def find_data_root(base_dir: Union[str, Path]) -> Path:
    """
    Walk base_dir up to 2 levels deep looking for a directory that contains
    both 'TRAIN' and 'TEST' subdirectories.

    Handles all common Kaggle unzip variants:
      data/TRAIN/
      data/images/TRAIN/
      data/dataset2-master/images/TRAIN/
      data/blood-cells/dataset2-master/images/TRAIN/

    Returns the resolved Path or raises FileNotFoundError.
    """
    base = Path(base_dir).resolve()

    def has_train_test(p: Path) -> bool:
        return (p / "TRAIN").is_dir() and (p / "TEST").is_dir()

    if has_train_test(base):
        return base

    for child in base.iterdir():
        if child.is_dir():
            if has_train_test(child):
                return child
            for grandchild in child.iterdir():
                if grandchild.is_dir():
                    if has_train_test(grandchild):
                        return grandchild
                    for great_grandchild in grandchild.iterdir():
                        if great_grandchild.is_dir() and has_train_test(great_grandchild):
                            return great_grandchild

    raise FileNotFoundError(
        f"Could not find TRAIN/ and TEST/ directories under '{base_dir}'. "
        f"Please check your data directory structure. "
        f"Contents of '{base_dir}': {[p.name for p in base.iterdir()]}"
    )


def get_transforms(split: str) -> transforms.Compose:
    """
    Returns the appropriate transform pipeline for the given split.

    split: 'train' | 'val' | 'test'

    Train: augmented pipeline to improve generalization.
    Val/Test: deterministic pipeline (resize + center crop + normalize).
    """
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )


class SubsetWithTransform(Dataset):
    """
    Wraps a torch Subset and applies a custom transform.

    This is needed because random_split returns Subsets that share the parent
    ImageFolder's transform. We use this to give the val split clean
    (non-augmented) transforms while the train split gets augmentation,
    even though both come from the same ImageFolder.
    """

    def __init__(self, subset: torch.utils.data.Subset, transform: transforms.Compose):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        # Load raw PIL image directly, bypassing parent's transform
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        img = self.subset.dataset.loader(path)
        return self.transform(img), label


def build_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    val_fraction: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple:
    """
    Build and return (train_loader, val_loader, test_loader).

    The TRAIN/ folder is split 80/20 into train and val using a fixed seed.
    The TEST/ folder is used as-is for final evaluation.

    Args:
        data_dir: Root directory containing TRAIN/ and TEST/ (or a parent).
        batch_size: Batch size for all loaders.
        val_fraction: Fraction of TRAIN used for validation.
        num_workers: DataLoader worker processes.
        seed: Random seed for reproducible split.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    root = find_data_root(data_dir)

    # Load the full TRAIN folder with a dummy transform (we override per-subset)
    train_full = datasets.ImageFolder(root=str(root / "TRAIN"))

    # Reproducible split
    n_total = len(train_full)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        train_full, [n_train, n_val], generator=generator
    )

    train_dataset = SubsetWithTransform(train_subset, get_transforms("train"))
    val_dataset = SubsetWithTransform(val_subset, get_transforms("val"))

    # TEST folder with test transform
    test_dataset = datasets.ImageFolder(
        root=str(root / "TEST"), transform=get_transforms("test")
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
