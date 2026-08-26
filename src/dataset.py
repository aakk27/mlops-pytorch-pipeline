"""CIFAR-10 data loading, transforms and DataLoader construction.

The transforms follow the standard CIFAR-10 recipe: random horizontal flips and
random crops with four pixels of padding for training, normalisation only for
validation. Normalisation constants are the per-channel means and standard
deviations of the CIFAR-10 training set.

``subset_fraction`` lets a run use a deterministic slice of the training data.
That is what makes a full Docker or Kubernetes demonstration finish in minutes
instead of hours, without changing any other part of the pipeline.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Per-channel statistics of the CIFAR-10 training split.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_transforms(train: bool = True) -> transforms.Compose:
    """Return the transform pipeline for the training or validation split."""
    if train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )


def _take_subset(dataset: torch.utils.data.Dataset, fraction: float, seed: int) -> Subset:
    """Deterministically sample ``fraction`` of ``dataset``.

    A fixed generator seed means two runs with the same configuration see the
    same examples, which keeps short demonstration runs comparable.
    """
    total = len(dataset)  # type: ignore[arg-type]
    keep = max(1, int(total * fraction))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator)[:keep].tolist()
    return Subset(dataset, indices)


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
    subset_fraction: float = 1.0,
    seed: int = 42,
    download: bool = True,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Build the CIFAR-10 training and validation loaders.

    Args:
        data_dir: Directory the dataset is read from or downloaded into. In the
            container this is a mounted volume, so the download happens once.
        batch_size: Examples per batch for both splits.
        num_workers: Worker processes per loader.
        subset_fraction: Fraction of each split to use, in ``(0, 1]``.
        seed: Seed controlling the subset sample.
        download: Whether torchvision may fetch the dataset if absent.
        pin_memory: Page-lock host memory to speed up host-to-device copies.
            Only meaningful for CUDA; MPS does not support it and warns, and on
            CPU there is no transfer to accelerate. The caller decides.

    Raises:
        ValueError: If ``subset_fraction`` is outside ``(0, 1]``.
    """
    if not 0.0 < subset_fraction <= 1.0:
        raise ValueError(f"subset_fraction must be in (0, 1], got {subset_fraction}")

    train_dataset: torch.utils.data.Dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=get_transforms(train=True),
    )
    val_dataset: torch.utils.data.Dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=get_transforms(train=False),
    )

    if subset_fraction < 1.0:
        train_dataset = _take_subset(train_dataset, subset_fraction, seed)
        val_dataset = _take_subset(val_dataset, subset_fraction, seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # Re-using workers avoids paying process start-up on every epoch.
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader
