"""Model definitions for CIFAR-10 image classification.

Two architectures are available:

``resnet18``
    ``torchvision``'s ResNet-18 with two adaptations for 32x32 inputs. The stock
    model is designed for 224x224 ImageNet images: its first convolution is a
    7x7 with stride 2 and it is followed by a 3x3 max-pool with stride 2, which
    together shrink the input by a factor of four before the residual stages
    even begin. On a 32x32 CIFAR image that destroys most of the spatial signal,
    so we swap in a 3x3 stride-1 convolution and drop the max-pool. This is the
    standard "CIFAR ResNet" adaptation and it is worth several accuracy points.

``simple_cnn``
    A small three-block convolutional network, useful as a fast baseline when
    iterating on the pipeline rather than on model quality.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18

# Canonical CIFAR-10 label order as produced by torchvision.datasets.CIFAR10.
CIFAR10_CLASSES: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

SUPPORTED_ARCHITECTURES: tuple[str, ...] = ("resnet18", "simple_cnn")


class SimpleCNN(nn.Module):
    """A compact CNN baseline for 32x32 three-channel images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
        )
        # Global average pooling keeps the classifier independent of input size.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def _build_cifar_resnet18(num_classes: int) -> nn.Module:
    """ResNet-18 re-stemmed for 32x32 inputs."""
    model = resnet18(weights=None, num_classes=num_classes)
    # 7x7/stride-2 stem -> 3x3/stride-1: preserve spatial detail on small images.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # The ImageNet max-pool would halve 32x32 to 16x16 before any residual block.
    model.maxpool = nn.Identity()
    return model


def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    """Return an initialised model for ``architecture``.

    Args:
        architecture: One of :data:`SUPPORTED_ARCHITECTURES`.
        num_classes: Number of output logits.

    Raises:
        ValueError: If ``architecture`` is not supported.
    """
    key = architecture.strip().lower()
    if key == "resnet18":
        return _build_cifar_resnet18(num_classes)
    if key == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    raise ValueError(
        f"Unsupported architecture {architecture!r}. Expected one of {SUPPORTED_ARCHITECTURES}."
    )


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters, handy for start-up logs."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
