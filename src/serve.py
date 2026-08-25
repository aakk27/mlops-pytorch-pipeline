"""FastAPI inference service for the trained CIFAR-10 classifier.

Endpoints
---------
``GET /health``
    Returns 200 once a checkpoint is loaded, 503 otherwise. Both the Kubernetes
    liveness and readiness probes point here. Because the serving Deployment can
    legitimately start before the training Job has written a checkpoint, the
    handler retries the load on each call rather than failing permanently: the
    pod turns ready by itself as soon as the checkpoint appears on the shared
    volume.

``POST /predict``
    Accepts a multipart image upload under the field name ``image`` and returns
    the predicted class together with the full probability distribution.

``GET /metadata``
    Reports what is loaded: architecture, validation metrics, class names.

The checkpoint path comes from ``MODEL_CHECKPOINT_PATH`` and defaults to the
location the training Job writes to on the shared PersistentVolumeClaim.
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from model import CIFAR10_CLASSES, get_model

CHECKPOINT_PATH = Path(os.getenv("MODEL_CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt"))

# Inference-time preprocessing. Uploads arrive at arbitrary sizes and colour
# modes, so we coerce to 32x32 RGB and apply the same normalisation used in
# training — a mismatch here silently wrecks accuracy.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
PREPROCESS = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ]
)


class ModelStore:
    """Holds the loaded model and the metadata that came with its checkpoint."""

    def __init__(self) -> None:
        self.model: torch.nn.Module | None = None
        self.class_names: list[str] = list(CIFAR10_CLASSES)
        self.metadata: dict[str, Any] = {}
        self.device = torch.device("cpu")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self, path: Path | None = None) -> bool:
        """Load a checkpoint. Returns True on success, False if unavailable.

        The path is resolved at call time rather than bound as a default
        argument, so the module-level configuration is read on every attempt
        rather than frozen at import.

        Never raises: a missing checkpoint is an expected state while the
        training Job is still running, and the health probe reports it.
        """
        target = Path(path) if path is not None else CHECKPOINT_PATH
        if not target.exists():
            return False
        try:
            # weights_only=True refuses to unpickle arbitrary objects, so a
            # tampered checkpoint file cannot execute code on load.
            checkpoint = torch.load(target, map_location=self.device, weights_only=True)
        except Exception:  # noqa: BLE001 - any failure means "not loaded yet"
            return False

        architecture = checkpoint.get("architecture", "resnet18")
        num_classes = int(checkpoint.get("num_classes", 10))
        model = get_model(architecture=architecture, num_classes=num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval().to(self.device)

        self.model = model
        self.class_names = list(checkpoint.get("class_names", CIFAR10_CLASSES))
        self.metadata = {
            "architecture": architecture,
            "num_classes": num_classes,
            "checkpoint_path": str(target),
            "trained_epochs": checkpoint.get("epoch"),
            "val_loss": checkpoint.get("val_loss"),
            "val_accuracy": checkpoint.get("val_accuracy"),
        }
        return True


store = ModelStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Attempt one load at start-up; absence is tolerated, not fatal."""
    store.load()
    yield


app = FastAPI(
    title="CIFAR-10 Classifier",
    description="Inference service for the PyTorch model trained by this pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> JSONResponse:
    """Readiness and liveness probe target."""
    if not store.is_loaded:
        # Retry the load: the checkpoint may have appeared since start-up.
        store.load()
    if store.is_loaded:
        return JSONResponse(status_code=200, content={"status": "ok", "model_loaded": True})
    return JSONResponse(
        status_code=503,
        content={
            "status": "unavailable",
            "model_loaded": False,
            "detail": f"No checkpoint at {CHECKPOINT_PATH}",
        },
    )


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    """Describe the loaded checkpoint."""
    if not store.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"class_names": store.class_names, **store.metadata}


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict[str, Any]:
    """Classify an uploaded image.

    Returns the top prediction plus the full probability distribution, so a
    caller can apply its own confidence threshold.
    """
    if not store.is_loaded and not store.load():
        raise HTTPException(status_code=503, detail=f"No checkpoint at {CHECKPOINT_PATH}")

    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file was empty")

    try:
        pil_image = Image.open(io.BytesIO(payload)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Not a readable image: {exc}") from exc

    tensor = PREPROCESS(pil_image).unsqueeze(0).to(store.device)

    with torch.no_grad():
        logits = store.model(tensor)  # type: ignore[misc]
        probabilities = F.softmax(logits, dim=1).squeeze(0)

    confidence, index = torch.max(probabilities, dim=0)
    return {
        "filename": image.filename,
        "predicted_class": store.class_names[int(index)],
        "confidence": round(float(confidence), 4),
        "probabilities": {
            name: round(float(p), 4)
            for name, p in zip(store.class_names, probabilities.tolist(), strict=False)
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=int(os.getenv("SERVE_PORT", "8080")),
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
