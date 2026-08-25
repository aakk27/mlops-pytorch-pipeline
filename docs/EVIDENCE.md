# Verification Evidence

Captured output proving each part of the pipeline works. Appended to as stages
complete; the Part F section supplies the terminal output the brief asks for in
the final PR.

---

## Stage 0 — Cluster provisioning

```
$ minikube start --cpus=4 --memory=7168 --driver=docker
😄  minikube v1.38.1 on Darwin 26.5.2 (arm64)
✨  Using the docker driver based on user configuration
📌  Using Docker Desktop driver with root privileges
👍  Starting "minikube" primary control-plane node in "minikube" cluster
🚜  Pulling base image v0.0.50 ...
🔥  Creating docker container (CPUs=4, Memory=7168MB) ...
🐳  Preparing Kubernetes v1.35.1 on Docker 29.2.1 ...
🔎  Verifying Kubernetes components...
🌟  Enabled addons: storage-provisioner, default-storageclass
🏄  Done! kubectl is now configured to use "minikube" cluster

$ kubectl get nodes
NAME       STATUS   ROLES           AGE   VERSION
minikube   Ready    control-plane   24s   v1.35.1
```

---

## Stage 1 — Model, training and serving

### Static analysis

```
$ ruff check src tests
All checks passed!

$ ruff format --check src tests
6 files already formatted
```

### Unit and API tests

```
$ pytest tests -q
......................                                              [100%]
22 passed in 4.37s
```

Coverage by area:

| Area | Tests |
|---|---|
| Forward-pass shapes across both architectures and 3 class counts | 5 |
| ResNet stem adaptation (kernel, stride, Identity max-pool, resolution) | 2 |
| Architecture factory (unknown rejected, case-insensitive lookup) | 2 |
| Weights actually update after an optimiser step | 1 |
| Class list and parameter counting | 2 |
| Transform pipeline shape, normalisation values, determinism | 3 |
| `/health` 503 without a checkpoint, 200 with one | 2 |
| `/metadata` contents | 1 |
| `/predict` distribution validity, resizing, 400s for bad uploads | 4 |

### Bug found by the test suite

`ModelStore.load()` was originally declared as:

```python
def load(self, path: Path = CHECKPOINT_PATH) -> bool:
```

A default argument is evaluated once, at import time, so the configured
checkpoint path was frozen at module load and later reconfiguration had no
effect. The API tests caught it immediately — patching the module constant did
not change the path the loader used. It now resolves at call time:

```python
def load(self, path: Path | None = None) -> bool:
    target = Path(path) if path is not None else CHECKPOINT_PATH
```

### End-to-end run

Training loop, early stopping, checkpoint writing and HTTP inference exercised
in one pass. Structured logs are emitted as JSON Lines on stdout:

```
{"event": "env_overrides_applied", "overrides": {"MAX_EPOCHS": 4, "BATCH_SIZE": 32, "SUBSET_FRACTION": 0.02, ...}}
{"event": "run_started", "config_path": "configs/training_config.yaml", "device": "cpu", "architecture": "resnet18", "num_classes": 10, "trainable_parameters": 11173962, "epochs": 4, "batch_size": 32, "learning_rate": 0.001, "subset_fraction": 0.02}
{"event": "data_ready", "train_batches": 8, "val_batches": 4, "train_examples": 256, "val_examples": 128}
{"epoch": 1, "train_loss": 2.4194, "train_accuracy": 0.1133, "val_loss": 2.8352, "val_accuracy": 0.1172, "epoch_seconds": 7.4}
{"event": "checkpoint_saved", "path": "checkpoints/classifier_v1.pt", "epoch": 1}
{"epoch": 2, "train_loss": 0.2572, "train_accuracy": 0.9883, "val_loss": 2.9568, "val_accuracy": 0.1094, "epoch_seconds": 5.2}
{"event": "no_improvement", "epoch": 2, "patience_counter": 1}
{"epoch": 3, "train_loss": 0.0096, "train_accuracy": 1.0, "val_loss": 2.7535, "val_accuracy": 0.1406, "epoch_seconds": 5.0}
{"event": "checkpoint_saved", "path": "checkpoints/classifier_v1.pt", "epoch": 3}
{"epoch": 4, "train_loss": 0.0038, "train_accuracy": 1.0, "val_loss": 2.6485, "val_accuracy": 0.1406, "epoch_seconds": 4.9}
{"event": "checkpoint_saved", "path": "checkpoints/classifier_v1.pt", "epoch": 4}
{"event": "training_complete", "best_val_loss": 2.6485, "best_val_accuracy": 0.1406, "checkpoint": "checkpoints/classifier_v1.pt", "total_seconds": 23.8}
```

> **Note on these numbers.** This run used *synthetic random tensors*, not
> CIFAR-10 — the dataset host was unreachable from the environment where the
> pipeline was first exercised. Training accuracy reaching 1.0 while validation
> accuracy stays near chance (0.14) is the expected outcome: the model memorises
> 256 random images and cannot generalise, because there is nothing to
> generalise to. The run proves the *mechanics* — config resolution, override
> handling, the training loop, checkpoint writing, early-stopping bookkeeping —
> not model quality. Real CIFAR-10 metrics are recorded in the Stage 2 and
> Stage 4 sections.

Checkpoint contents:

```
checkpoint keys: ['architecture', 'class_names', 'epoch', 'model_state_dict',
                  'num_classes', 'optimizer_state_dict', 'val_accuracy', 'val_loss']
```

Serving the resulting checkpoint over HTTP:

```
GET /health   -> 200 {'status': 'ok', 'model_loaded': True}
GET /metadata -> architecture=resnet18 trained_epochs=4 val_accuracy=0.140625
POST /predict -> 200
{
  "filename": "test_image.png",
  "predicted_class": "horse",
  "confidence": 0.6335,
  "probabilities": {
    "airplane": 0.085,  "automobile": 0.0004, "bird": 0.043,  "cat": 0.0477,
    "deer": 0.0087,     "dog": 0.0576,        "frog": 0.0159, "horse": 0.6335,
    "ship": 0.0987,     "truck": 0.0095
  }
}
```

Probabilities sum to 1.0, confirming the softmax and the class-name mapping are
wired correctly.

### Malformed override handling

```
{"event": "invalid_env_override", "variable": "LEARNING_RATE", "value": "not-a-number"}
```

The run continued with the YAML value rather than crashing.

---

## Stage 1b — Real CIFAR-10 run

Run on the target machine (Apple Silicon Mac mini, Python 3.13 virtualenv),
5% of the dataset for 2 epochs:

```
$ pytest tests -v
22 passed, 1 warning in 12.65s

$ TRAINING_CONFIG_PATH=configs/training_config.yaml DATA_DIR=./data \
  CHECKPOINT_DIR=./checkpoints SUBSET_FRACTION=0.05 MAX_EPOCHS=2 \
  NUM_WORKERS=0 python src/train.py

{"event": "env_overrides_applied", "overrides": {"MAX_EPOCHS": 2, "NUM_WORKERS": 0, "SUBSET_FRACTION": 0.05, "DATA_DIR": "./data", "CHECKPOINT_DIR": "./checkpoints"}}
{"event": "run_started", "config_path": "configs/training_config.yaml", "device": "mps", "architecture": "resnet18", "num_classes": 10, "trainable_parameters": 11173962, "epochs": 2, "batch_size": 64, "learning_rate": 0.001, "subset_fraction": 0.05}
100.0%
{"event": "data_ready", "train_batches": 40, "val_batches": 8, "train_examples": 2500, "val_examples": 500}
{"epoch": 1, "train_loss": 2.0321, "train_accuracy": 0.256, "val_loss": 2.2223, "val_accuracy": 0.28, "epoch_seconds": 10.1}
{"event": "checkpoint_saved", "path": "checkpoints/classifier_v1.pt", "epoch": 1}
{"epoch": 2, "train_loss": 1.7614, "train_accuracy": 0.356, "val_loss": 2.093, "val_accuracy": 0.312, "epoch_seconds": 4.7}
{"event": "checkpoint_saved", "path": "checkpoints/classifier_v1.pt", "epoch": 2}
{"event": "training_complete", "best_val_loss": 2.093, "best_val_accuracy": 0.312, "checkpoint": "checkpoints/classifier_v1.pt", "total_seconds": 2125.2}
```

### Reading these numbers

| Observation | Interpretation |
|---|---|
| `device: mps` | Apple Metal GPU selected. Containers will report `cpu` — see D-017 |
| 2500 train / 500 val examples | `subset_fraction: 0.05` applied to both splits as intended |
| val_accuracy 0.28 -> 0.312 | Genuine learning. Chance is 0.10 for ten balanced classes, so the model is roughly 3x better than random after two epochs on 5% of the data |
| val_loss 2.2223 -> 2.093 | Decreasing, so early stopping correctly did not trigger |
| `epoch_seconds` 10.1 then 4.7 | The first epoch pays one-off cost — CUDA/MPS kernel compilation and page-cache warming. The steady-state figure is ~4.7s |
| `total_seconds: 2125.2` | **Misleading in this run.** ~2,110s of it was the one-time CIFAR-10 download from `www.cs.toronto.edu`, not compute |

The last row prompted a fix (D-018): timing is now reported as three separate
fields so download time can never again be read as training time. The same run
today would report approximately `training_seconds: 15`, `data_seconds: 2110`,
`total_seconds: 2125`.

A warning also surfaced and was fixed (D-019):

```
UserWarning: 'pin_memory' argument is set as true but not supported on MPS now,
device pinned memory won't be used.
```

### Post-fix verification

Same command re-run on the target machine after the timing and `pin_memory`
fixes, with the dataset already cached:

```
$ TRAINING_CONFIG_PATH=configs/training_config.yaml DATA_DIR=./data \
  CHECKPOINT_DIR=./checkpoints SUBSET_FRACTION=0.05 MAX_EPOCHS=2 \
  NUM_WORKERS=0 python src/train.py 2>&1 | tee run.jsonl | jq -Rc 'fromjson? // .'

{"event":"env_overrides_applied","overrides":{"MAX_EPOCHS":2,"NUM_WORKERS":0,"SUBSET_FRACTION":0.05,"DATA_DIR":"./data","CHECKPOINT_DIR":"./checkpoints"}}
{"event":"run_started","config_path":"configs/training_config.yaml","device":"mps","architecture":"resnet18","num_classes":10,"trainable_parameters":11173962,"epochs":2,"batch_size":64,"learning_rate":0.001,"subset_fraction":0.05}
{"event":"data_ready","train_batches":40,"val_batches":8,"train_examples":2500,"val_examples":500,"data_seconds":0.9}
{"epoch":1,"train_loss":2.0321,"train_accuracy":0.256,"val_loss":2.2223,"val_accuracy":0.28,"epoch_seconds":5.4}
{"event":"checkpoint_saved","path":"checkpoints/classifier_v1.pt","epoch":1}
{"epoch":2,"train_loss":1.7614,"train_accuracy":0.356,"val_loss":2.093,"val_accuracy":0.312,"epoch_seconds":4.7}
{"event":"checkpoint_saved","path":"checkpoints/classifier_v1.pt","epoch":2}
{"event":"training_complete","best_val_loss":2.093,"best_val_accuracy":0.312,"checkpoint":"checkpoints/classifier_v1.pt","training_seconds":10.4,"data_seconds":0.9,"total_seconds":11.5}
```

The log is valid JSON Lines end to end — piping it through `jq -Rc 'fromjson?'`
parses every record, which is the property that makes it usable by a log shipper
or by `kubectl logs ... | jq` in the cluster.

### What this run demonstrates

**Reproducibility.** Every metric is identical to the previous run — the same
`train_loss` of 2.0321, the same `val_accuracy` of 0.28, the same values at
epoch 2. Two independent invocations producing identical numbers confirms that
`set_seed()` and the seeded subset sampling (D-006) genuinely control every
source of randomness in the pipeline. This is a stronger claim than the code
merely containing seeding calls.

**Timing is now honest.** `training_seconds: 10.4` against `data_seconds: 0.9`
and `total_seconds: 11.5`. The same work previously reported `total_seconds:
2125.2` because the one-off dataset download was folded into it (D-018).

**The first-epoch cost was warm-up, not the model.** Epoch 1 fell from 10.1s to
5.4s while epoch 2 stayed at 4.7s. The difference is MPS kernel compilation and
page-cache warming on the first run, now paid off. Steady-state is ~4.7s per
epoch on 2,500 images.

**No warnings.** The `pin_memory` UserWarning is gone (D-019).

### Baseline for sizing the containerised run

| Measure | Value |
|---|---|
| Steady-state epoch, 2500 images, MPS | ~4.7 s |
| Dataset load from cache | ~0.9 s |
| Validation accuracy after 2 epochs | 0.312 (chance = 0.10) |

Container and Kubernetes runs fall back to CPU (D-017), so these figures are a
lower bound on containerised time, not a prediction. The Job's demonstration
parameters are chosen from a measured containerised epoch instead.

---

## Stage 2 — Docker

*(To be filled: image builds, sizes, non-root verification, HEALTHCHECK
inspection, containerised training run, containerised prediction.)*

```
```

---

## Stage 3 — Kubernetes manifests

*(To be filled: `kubectl apply` output, `kubectl explain`/dry-run validation.)*

```
```

---

## Stage 4 — End-to-end on the cluster (Part F)

*(To be filled: namespace/configmap/PVC/Job apply, Job logs, Job completion,
serving Deployment rollout, pod status, `describe deployment`, HPA status,
port-forward, `curl /predict` response.)*

```
```
