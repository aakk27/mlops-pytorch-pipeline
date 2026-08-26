"""Unit tests for the model definitions and data transforms."""

from __future__ import annotations

import pytest
import torch
from PIL import Image
from torch import nn

from dataset import CIFAR10_MEAN, CIFAR10_STD, get_transforms
from model import (
    CIFAR10_CLASSES,
    SUPPORTED_ARCHITECTURES,
    SimpleCNN,
    count_parameters,
    get_model,
)


@pytest.mark.parametrize("architecture", SUPPORTED_ARCHITECTURES)
def test_forward_pass_returns_logits_per_class(architecture: str) -> None:
    """Every supported architecture maps a batch of images to one logit per class."""
    model = get_model(architecture=architecture, num_classes=10)
    model.eval()
    batch = torch.randn(4, 3, 32, 32)

    with torch.no_grad():
        output = model(batch)

    assert output.shape == (4, 10)
    assert torch.isfinite(output).all(), "Forward pass produced NaN or inf"


@pytest.mark.parametrize("num_classes", [2, 10, 100])
def test_output_width_follows_num_classes(num_classes: int) -> None:
    model = get_model(architecture="simple_cnn", num_classes=num_classes)
    model.eval()

    with torch.no_grad():
        output = model(torch.randn(2, 3, 32, 32))

    assert output.shape == (2, num_classes)


def test_resnet18_stem_is_adapted_for_cifar() -> None:
    """The ImageNet stem must be replaced, or 32x32 inputs lose most detail.

    Stock ResNet-18 starts with a 7x7 stride-2 convolution followed by a stride-2
    max-pool. This project swaps in a 3x3 stride-1 convolution and removes the
    pool; these assertions stop that adaptation being silently reverted.
    """
    model = get_model(architecture="resnet18", num_classes=10)

    assert isinstance(model.conv1, nn.Conv2d)
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, nn.Identity)


def test_resnet18_preserves_spatial_resolution_into_first_stage() -> None:
    """After the adapted stem a 32x32 input should still be 32x32."""
    model = get_model(architecture="resnet18", num_classes=10)
    model.eval()

    with torch.no_grad():
        stem_output = model.maxpool(model.relu(model.bn1(model.conv1(torch.randn(1, 3, 32, 32)))))

    assert stem_output.shape[-2:] == (32, 32)


def test_unknown_architecture_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported architecture"):
        get_model(architecture="vgg16", num_classes=10)


def test_architecture_lookup_is_case_insensitive() -> None:
    assert isinstance(get_model(architecture="SimPle_CNN", num_classes=10), SimpleCNN)


def test_model_is_trainable() -> None:
    """A single optimiser step must actually change the weights."""
    torch.manual_seed(0)
    model = get_model(architecture="simple_cnn", num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    before = next(model.parameters()).detach().clone()

    loss = criterion(model(torch.randn(4, 3, 32, 32)), torch.randint(0, 10, (4,)))
    loss.backward()
    optimizer.step()

    after = next(model.parameters()).detach()
    assert not torch.allclose(before, after), "Weights did not update after an optimiser step"


def test_count_parameters_is_positive() -> None:
    assert count_parameters(get_model("resnet18", 10)) > 0


def test_cifar10_class_list_is_complete() -> None:
    assert len(CIFAR10_CLASSES) == 10
    assert len(set(CIFAR10_CLASSES)) == 10
    assert CIFAR10_CLASSES[0] == "airplane"


def test_training_transform_produces_normalised_tensor() -> None:
    """Training transforms must yield a 3x32x32 tensor from a larger image."""
    image = Image.new("RGB", (64, 64), color=(120, 130, 140))

    tensor = get_transforms(train=True)(image)

    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32


def test_validation_transform_applies_expected_normalisation() -> None:
    """A flat white image maps to (1 - mean) / std on each channel."""
    image = Image.new("RGB", (32, 32), color=(255, 255, 255))

    tensor = get_transforms(train=False)(image)

    expected = torch.tensor([(1.0 - m) / s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD, strict=True)])
    channel_means = tensor.mean(dim=(1, 2))
    assert torch.allclose(channel_means, expected, atol=1e-4)


def test_validation_transform_is_deterministic() -> None:
    """No augmentation on the validation path: repeated calls must agree."""
    image = Image.new("RGB", (32, 32), color=(10, 200, 90))
    transform = get_transforms(train=False)

    assert torch.equal(transform(image), transform(image))
