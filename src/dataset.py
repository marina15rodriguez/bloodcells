from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

CLASSES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 240


def find_data_root(data_dir: Path) -> Path:
    for train_dir in Path(data_dir).rglob("TRAIN"):
        if train_dir.is_dir() and (train_dir.parent / "TEST").is_dir():
            return train_dir.parent
    raise FileNotFoundError(f"Could not find TRAIN/TEST dirs under {data_dir}")


class SubsetWithTransform(Dataset):
    """Wraps a Subset to apply a different transform than the parent dataset."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(split: str) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def build_dataloaders(data_dir, batch_size=32, num_workers=2, seed=42):
    root = find_data_root(Path(data_dir))

    # Load TRAIN with no transform so we can apply different ones to train/val subsets
    train_full = datasets.ImageFolder(root=str(root / "TRAIN"), transform=None)

    n_train = int(0.8 * len(train_full))
    n_val = len(train_full) - n_train
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(train_full, [n_train, n_val], generator=generator)

    train_ds = SubsetWithTransform(train_subset, get_transforms("train"))
    val_ds = SubsetWithTransform(val_subset, get_transforms("val"))
    test_ds = datasets.ImageFolder(root=str(root / "TEST"), transform=get_transforms("test"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
