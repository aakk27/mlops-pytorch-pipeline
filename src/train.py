"""Training entry point.

Reads hyperparameters from a YAML configuration file, trains a CIFAR-10
classifier, emits one JSON object per epoch on stdout, and writes a checkpoint
whenever validation loss improves. Training stops early once validation loss has
failed to improve for ``early_stopping_patience`` consecutive epochs.

Configuration resolution order (first hit wins):

1. ``TRAINING_CONFIG_PATH`` environment variable
2. ``/app/configs/training_config.yaml`` (the ConfigMap mount inside the cluster)
3. ``configs/training_config.yaml`` relative to the repository root

Individual values can then be overridden by environment variables, which is how
the container and the Kubernetes Job run a short demonstration without editing
the committed configuration:

===========================  ====================================
Environment variable         Overrides
===========================  ====================================
``MAX_EPOCHS``               ``training.epochs``
``BATCH_SIZE``               ``training.batch_size``
``LEARNING_RATE``            ``training.learning_rate``
``NUM_WORKERS``              ``training.num_workers``
``SUBSET_FRACTION``          ``data.subset_fraction``
``DATA_DIR``                 ``data.data_dir``
``CHECKPOINT_DIR``           ``output.checkpoint_dir``
===========================  ====================================

Structured logs are written to stdout as JSON Lines so that ``kubectl logs``
output can be piped straight into ``jq`` or a log shipper.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from dataset import get_dataloaders
from model import CIFAR10_CLASSES, count_parameters, get_model

CONFIG_SEARCH_PATHS = (
    Path("/app/configs/training_config.yaml"),
    Path("configs/training_config.yaml"),
    Path(__file__).resolve().parent.parent / "configs" / "training_config.yaml",
)

# Environment variable -> (config section, key, caster)
ENV_OVERRIDES: tuple[tuple[str, str, str, Any], ...] = (
    ("MAX_EPOCHS", "training", "epochs", int),
    ("BATCH_SIZE", "training", "batch_size", int),
    ("LEARNING_RATE", "training", "learning_rate", float),
    ("NUM_WORKERS", "training", "num_workers", int),
    ("SUBSET_FRACTION", "data", "subset_fraction", float),
    ("DATA_DIR", "data", "data_dir", str),
    ("CHECKPOINT_DIR", "output", "checkpoint_dir", str),
)


def log(**fields: Any) -> None:
    """Emit one JSON object per line on stdout, flushed immediately.

    Flushing matters in containers: without it Python buffers stdout when it is
    not a TTY and ``kubectl logs`` shows nothing until the process exits.
    """
    print(json.dumps(fields), flush=True)


def resolve_config_path() -> Path:
    """Locate the training configuration file."""
    override = os.getenv("TRAINING_CONFIG_PATH")
    if override:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(f"TRAINING_CONFIG_PATH points at a missing file: {path}")
        return path
    for candidate in CONFIG_SEARCH_PATHS:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No training configuration found. Looked in: "
        + ", ".join(str(p) for p in CONFIG_SEARCH_PATHS)
    )


def load_config(config_path: str | Path) -> dict:
    """Parse the YAML configuration file."""
    with open(config_path) as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration at {config_path} did not parse to a mapping.")
    return config


def apply_env_overrides(config: dict) -> dict:
    """Overlay environment variables onto the parsed configuration."""
    applied: dict[str, Any] = {}
    for env_name, section, key, cast in ENV_OVERRIDES:
        raw = os.getenv(env_name)
        if raw is None or raw == "":
            continue
        try:
            value = cast(raw)
        except (TypeError, ValueError):
            log(event="invalid_env_override", variable=env_name, value=raw)
            continue
        config.setdefault(section, {})[key] = value
        applied[env_name] = value
    if applied:
        log(event="env_overrides_applied", overrides=applied)
    return config


def set_seed(seed: int) -> None:
    """Seed every source of randomness we control, for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    """Pick the best available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # getattr keeps this working on builds without the MPS backend compiled in.
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training pass and return ``(average_loss, accuracy)``."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Weight by batch size so the final average is over examples, not batches
        # (the last batch is usually smaller than the rest).
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate on the validation split and return ``(average_loss, accuracy)``."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / total, correct / total


def main() -> int:
    started = time.time()

    config_path = resolve_config_path()
    config = apply_env_overrides(load_config(config_path))

    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    out_cfg = config["output"]

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    device = select_device()
    architecture = model_cfg["architecture"]
    num_classes = int(model_cfg["num_classes"])

    model = get_model(architecture=architecture, num_classes=num_classes).to(device)

    log(
        event="run_started",
        config_path=str(config_path),
        device=str(device),
        architecture=architecture,
        num_classes=num_classes,
        trainable_parameters=count_parameters(model),
        epochs=int(train_cfg["epochs"]),
        batch_size=int(train_cfg["batch_size"]),
        learning_rate=float(train_cfg["learning_rate"]),
        subset_fraction=float(data_cfg.get("subset_fraction", 1.0)),
    )

    data_started = time.time()
    train_loader, val_loader = get_dataloaders(
        data_dir=data_cfg["data_dir"],
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg.get("num_workers", 2)),
        subset_fraction=float(data_cfg.get("subset_fraction", 1.0)),
        seed=seed,
        # Pinned host memory only accelerates CUDA transfers. MPS does not
        # support it and warns on every loader; on CPU there is no transfer.
        pin_memory=device.type == "cuda",
    )
    data_seconds = time.time() - data_started
    log(
        event="data_ready",
        train_batches=len(train_loader),
        val_batches=len(val_loader),
        train_examples=len(train_loader.dataset),  # type: ignore[arg-type]
        val_examples=len(val_loader.dataset),  # type: ignore[arg-type]
        # Dominated by the one-off dataset download on a cold volume. Reported
        # separately so it is never mistaken for training time.
        data_seconds=round(data_seconds, 1),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["learning_rate"]))
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    patience_counter = 0
    patience = int(train_cfg["early_stopping_patience"])

    checkpoint_dir = Path(out_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / out_cfg["model_name"]

    epochs = int(train_cfg["epochs"])
    training_started = time.time()
    for epoch in range(1, epochs + 1):
        epoch_started = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        log(
            epoch=epoch,
            train_loss=round(train_loss, 4),
            train_accuracy=round(train_acc, 4),
            val_loss=round(val_loss, 4),
            val_accuracy=round(val_acc, 4),
            epoch_seconds=round(time.time() - epoch_started, 1),
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_acc
            patience_counter = 0
            # The checkpoint carries everything the serving container needs to
            # rebuild the model, so serve.py never has to guess the architecture.
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "architecture": architecture,
                    "num_classes": num_classes,
                    "class_names": list(CIFAR10_CLASSES),
                },
                save_path,
            )
            log(event="checkpoint_saved", path=str(save_path), epoch=epoch)
        else:
            patience_counter += 1
            log(event="no_improvement", epoch=epoch, patience_counter=patience_counter)
            if patience_counter >= patience:
                log(event="early_stopping", epoch=epoch, patience=patience)
                break

    log(
        event="training_complete",
        best_val_loss=round(best_val_loss, 4),
        best_val_accuracy=round(best_val_accuracy, 4),
        checkpoint=str(save_path),
        # training_seconds covers the epoch loop alone; total_seconds is
        # wall-clock for the whole process and includes dataset preparation.
        training_seconds=round(time.time() - training_started, 1),
        data_seconds=round(data_seconds, 1),
        total_seconds=round(time.time() - started, 1),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
