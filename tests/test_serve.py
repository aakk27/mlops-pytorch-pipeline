"""API tests for the FastAPI inference service.

These exercise the contract the Kubernetes probes and the assignment's ``curl``
commands depend on: /health flips from 503 to 200 once a checkpoint exists, and
/predict returns a well-formed probability distribution.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from model import CIFAR10_CLASSES, get_model


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    """Write a small but genuine checkpoint in the format train.py produces."""
    model = get_model(architecture="simple_cnn", num_classes=10)
    path = tmp_path / "classifier_v1.pt"
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "val_loss": 1.23,
            "val_accuracy": 0.45,
            "architecture": "simple_cnn",
            "num_classes": 10,
            "class_names": list(CIFAR10_CLASSES),
        },
        path,
    )
    return path


@pytest.fixture
def client(checkpoint: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A client whose service points at the freshly written checkpoint."""
    import serve

    monkeypatch.setattr(serve, "CHECKPOINT_PATH", checkpoint)
    serve.store.model = None  # force a reload against the patched path
    with TestClient(serve.app) as test_client:
        yield test_client


def sample_image_bytes(size: tuple[int, int] = (32, 32)) -> bytes:
    """An in-memory PNG to upload."""
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(90, 140, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_reports_503_without_a_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Before training has produced a checkpoint the pod must not report ready."""
    import serve

    monkeypatch.setattr(serve, "CHECKPOINT_PATH", Path("/nonexistent/classifier_v1.pt"))
    serve.store.model = None

    with TestClient(serve.app) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["model_loaded"] is False


def test_health_reports_200_once_loaded(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_metadata_describes_the_checkpoint(client: TestClient) -> None:
    body = client.get("/metadata").json()

    assert body["architecture"] == "simple_cnn"
    assert body["num_classes"] == 10
    assert body["class_names"] == list(CIFAR10_CLASSES)
    assert body["val_accuracy"] == 0.45


def test_predict_returns_a_valid_distribution(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"image": ("test_image.png", sample_image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in CIFAR10_CLASSES
    assert len(body["probabilities"]) == 10
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_resizes_non_cifar_dimensions(client: TestClient) -> None:
    """Arbitrary upload sizes must be accepted, not just 32x32."""
    response = client.post(
        "/predict",
        files={"image": ("big.png", sample_image_bytes((256, 180)), "image/png")},
    )

    assert response.status_code == 200


def test_predict_rejects_a_non_image_upload(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"image": ("notes.txt", b"this is not an image", "text/plain")},
    )

    assert response.status_code == 400


def test_predict_rejects_an_empty_upload(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"image": ("empty.png", b"", "image/png")},
    )

    assert response.status_code == 400
