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

### Image builds

Both images built on the first attempt. The dependency install took 65 seconds,
which confirms the `linux/arm64` CPU wheel resolved from the PyTorch index
rather than falling back to a CUDA-enabled wheel from PyPI:

```
=> [builder 3/5] RUN python -m venv /opt/venv                              1.8s
=> [builder 4/5] COPY requirements/train.txt ./requirements/train.txt      0.0s
=> [builder 5/5] RUN pip install --no-cache-dir -r requirements/train.txt 65.4s
=> [training 3/6] COPY --from=builder /opt/venv /opt/venv                  1.3s
=> [training 4/6] COPY src/ ./src/                                         0.1s
=> [training 5/6] COPY configs/ ./configs/                                 0.0s
=> [training 6/6] RUN mkdir -p /app/data /app/checkpoints                  0.1s
=> naming to docker.io/library/mlops-train:v1
```

The layer ordering is visible here: requirements are copied and installed before
any source, so editing `src/` re-runs only the last three sub-second steps.

### Image sizes

```
$ docker images | grep mlops
mlops-serve:v1   461bb84eb7ee   1.43GB   297MB
mlops-train:v1   0884567346b7   1.39GB   286MB
```

The serving image is **larger** than the training image. Both carry `torch` and
`torchvision`, which dominate; serving adds a web stack on top while training
adds only `numpy` and `PyYAML`. The requirement — inference dependencies only,
no training libraries — is satisfied, but the image is not smaller in absolute
terms. See D-020 for the full reasoning and the options considered.

The second column is the compressed size, which is what actually transfers on a
pull and therefore what governs pod start-up latency in the cluster.

### C7 — Non-root user

```
$ docker run --rm mlops-serve:v1 id
uid=1001(appuser) gid=1001(appgroup) groups=1001(appgroup)
```

### C8 — HEALTHCHECK declared

```
$ docker inspect --format='{{json .Config.Healthcheck}}' mlops-serve:v1
{"Test":["CMD-SHELL","python -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4)\" || exit 1"],
 "Interval":30000000000,"Timeout":5000000000,"StartPeriod":15000000000,"Retries":3}
```

### C8 — HEALTHCHECK actually passing

Declaring a healthcheck and having it succeed are different claims. After the
start period elapsed:

```
$ docker ps --format '{{.Image}}\t{{.Status}}'
mlops-serve:v1                        Up About a minute (healthy)
gcr.io/k8s-minikube/kicbase:v0.0.50   Up 3 hours
```

`(healthy)` confirms the stdlib-`urllib` probe runs inside a slim image with no
curl installed, reaches `/health`, and gets a 200.

### C6 — Exposed port

```
$ docker inspect --format='{{json .Config.ExposedPorts}}' mlops-serve:v1
{"8080/tcp":{}}
```

### C5 — Training modules absent from the serving image

```
$ docker run --rm mlops-serve:v1 ls -la /app/src
total 20
drwxr-xr-x 1 appuser appgroup 4096 .
drwxr-xr-x 1 appuser appgroup 4096 ..
-rw-r--r-- 1 appuser appgroup 3795 model.py
-rw-r--r-- 1 appuser appgroup 6813 serve.py
```

Only the two modules inference imports. `train.py` and `dataset.py` are not in
the image.

### Containerised training run

```
$ docker run --rm \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    -e SUBSET_FRACTION=0.05 -e MAX_EPOCHS=2 -e NUM_WORKERS=2 \
    mlops-train:v1

{"event": "env_overrides_applied", "overrides": {"MAX_EPOCHS": 2, "NUM_WORKERS": 2, "SUBSET_FRACTION": 0.05}}
{"event": "run_started", "config_path": "/app/configs/training_config.yaml", "device": "cpu", "architecture": "resnet18", "num_classes": 10, "trainable_parameters": 11173962, "epochs": 2, "batch_size": 64, "learning_rate": 0.001, "subset_fraction": 0.05}
{"event": "data_ready", "train_batches": 40, "val_batches": 8, "train_examples": 2500, "val_examples": 500, "data_seconds": 2.0}
{"epoch": 1, "train_loss": 2.0292, "train_accuracy": 0.2632, "val_loss": 1.9178, "val_accuracy": 0.296, "epoch_seconds": 50.8}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt", "epoch": 1}
{"epoch": 2, "train_loss": 1.7498, "train_accuracy": 0.3384, "val_loss": 2.3598, "val_accuracy": 0.256, "epoch_seconds": 52.6}
{"event": "no_improvement", "epoch": 2, "patience_counter": 1}
{"event": "training_complete", "best_val_loss": 1.9178, "best_val_accuracy": 0.296, "checkpoint": "/app/checkpoints/classifier_v1.pt", "training_seconds": 103.5, "data_seconds": 2.0, "total_seconds": 105.7}
```

Three things this establishes.

**C3 — configuration resolution inside the container.** `config_path` is
`/app/configs/training_config.yaml`, the baked-in copy. In Kubernetes the
ConfigMap mounts over exactly that path, which is the mechanism Part D relies
on. The environment overrides were applied on top, as logged.

**B6 — early stopping demonstrated on real data.** Epoch 2's validation loss
*regressed* (1.9178 to 2.3598), so no checkpoint was written and the patience
counter incremented. `training_complete` reports epoch 1's 1.9178 as the best.
The checkpoint left on disk is therefore the better model, not the most recent
one — which is the entire point of the mechanism. Earlier runs had only ever
improved, so this is the first time the branch was exercised outside the tests.

**D-017 confirmed by measurement.** The container reports `device: cpu`, as
predicted: Docker on macOS runs in a Linux VM with no Metal access.

| Environment | Device | epoch_seconds | Ratio |
|---|---|---:|---:|
| Local virtualenv | mps | ~4.7 | 1x |
| Container | cpu | ~52 | ~11x |

This 11x factor — not the local MPS figure — is what sizes the Kubernetes Job.
At 2 CPUs the Job will be slower still, so `SUBSET_FRACTION=0.05, MAX_EPOCHS=2`
should land around three to four minutes.

**On the small metric differences.** The containerised `train_loss` of 2.0292
differs from the local 2.0321. This is `NUM_WORKERS=2` versus `0`, not
nondeterminism — see D-021.

### Serving the containerised checkpoint

A *separate* container, sharing only the checkpoint volume:

```
$ docker run --rm -p 8080:8080 -v "$(pwd)/checkpoints:/app/checkpoints" mlops-serve:v1

$ curl -s localhost:8080/health | jq
{
  "status": "ok",
  "model_loaded": true
}

$ curl -s localhost:8080/metadata | jq
{
  "class_names": ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"],
  "architecture": "resnet18",
  "num_classes": 10,
  "checkpoint_path": "/app/checkpoints/classifier_v1.pt",
  "trained_epochs": 1,
  "val_loss": 1.9177873344421388,
  "val_accuracy": 0.296
}
```

`trained_epochs: 1` is the important field. The serving container independently
reports the epoch-1 weights — the ones early stopping preserved when epoch 2
regressed. Two separate containers agree on which model won, communicating only
through the shared volume.

`architecture` and `class_names` came out of the checkpoint, not from any
configuration the serving image reads. That is D-007 working: `serve.py` was
never told what to rebuild.

### Prediction

```
$ python -c "...save CIFAR-10 test image 0 resized to 128x128..."
true label: cat

$ curl -s -X POST localhost:8080/predict -F "image=@test_image.png" | jq
{
  "filename": "test_image.png",
  "predicted_class": "cat",
  "confidence": 0.2227,
  "probabilities": {
    "airplane": 0.0774, "automobile": 0.0124, "bird": 0.1779, "cat": 0.2227,
    "deer": 0.2091,     "dog": 0.1492,        "frog": 0.0588, "horse": 0.0076,
    "ship": 0.0724,     "truck": 0.0125
  }
}
```

Correct — predicted `cat`, true label `cat`. The probabilities sum to exactly
1.0000.

The confidence of 0.2227 is low, with `deer` (0.2091) and `bird` (0.1779) close
behind. That is the appropriate shape for a model trained on 5% of CIFAR-10 for
two epochs at 29.6% validation accuracy: right on this example, but genuinely
uncertain. A 2-epoch model returning 0.99 confidence would indicate something
wrong with the softmax or the preprocessing.

The upload was 128x128 while the model takes 32x32, so this also exercises the
resize path in `PREPROCESS`.

### Part C requirement coverage

| Req | Claim | Evidence |
|---|---|---|
| C1 | Multi-stage training Dockerfile | Build log: `builder` then `training` stage |
| C2 | Dependency versions pinned | `requirements/*.txt`; 65s install |
| C3 | Config from mounted volume or env | `config_path: /app/configs/...` plus applied overrides |
| C4 | Slim Python base | `python:3.11-slim` in both stages |
| C5 | Inference dependencies only | `/app/src` holds only `model.py`, `serve.py` |
| C6 | Port 8080 exposed | `{"8080/tcp":{}}` |
| C7 | Non-root user | `uid=1001(appuser)` |
| C8 | HEALTHCHECK | Declared, and container reports `(healthy)` |
| C9 | Local build/run verification | Full chain: build, train, checkpoint, serve, predict |

---

## Stage 3 — Kubernetes manifests

Eight manifests, checked against every numeric requirement in Parts D and E
before being applied — resource requests and limits, probe periods and
thresholds, replica count, rolling update parameters, the read-only mount, and
the Service port mapping. All passed.

The cluster run that followed is recorded in Stage 4.

---

## Stage 4 — End-to-end on the cluster (Part F)

Full capture: `docs/logs/part-f-validation.txt`.
Cold first run: `docs/logs/k8s-cold-run.txt`.
Probe before/after: `docs/logs/startupprobe-before-after.txt`.

### Apply and train

```
$ kubectl apply -f k8s/namespace.yaml
$ kubectl apply -f k8s/configmap.yaml
$ kubectl apply -f k8s/pvc.yaml
$ kubectl apply -f k8s/training-job.yaml
job.batch/model-training created

$ kubectl get pvc -n ml-training
NAME             STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS
checkpoint-pvc   Bound    pvc-73785a07-3e84-4301-a7c8-ffc210c6b937   1Gi        RWO            standard
dataset-pvc      Bound    pvc-ea46f925-17b8-4437-9661-1990efcbb3f5   2Gi        RWO            standard

$ kubectl wait --for=condition=complete job/model-training -n ml-training
job.batch/model-training condition met

$ kubectl get job/model-training -n ml-training
NAME             STATUS     COMPLETIONS   DURATION   AGE
model-training   Complete   1/1           8m42s      8m42s
```

### Training logs from the cluster

```
{"event": "env_overrides_applied", "overrides": {"MAX_EPOCHS": 2, "NUM_WORKERS": 2, "SUBSET_FRACTION": 0.05}}
{"event": "run_started", "config_path": "/app/configs/training_config.yaml", "device": "cpu", "architecture": "resnet18", "num_classes": 10, "trainable_parameters": 11173962, "epochs": 2, "batch_size": 64, "learning_rate": 0.001, "subset_fraction": 0.05}
{"event": "data_ready", "train_batches": 40, "val_batches": 8, "train_examples": 2500, "val_examples": 500, "data_seconds": 1.4}
{"epoch": 1, "train_loss": 2.0292, "train_accuracy": 0.2632, "val_loss": 1.9178, "val_accuracy": 0.296, "epoch_seconds": 258.8}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt", "epoch": 1}
{"epoch": 2, "train_loss": 1.7498, "train_accuracy": 0.3384, "val_loss": 2.3598, "val_accuracy": 0.256, "epoch_seconds": 257.2}
{"event": "no_improvement", "epoch": 2, "patience_counter": 1}
{"event": "training_complete", "best_val_loss": 1.9178, "best_val_accuracy": 0.296, "checkpoint": "/app/checkpoints/classifier_v1.pt", "training_seconds": 516.1, "data_seconds": 1.4, "total_seconds": 517.6}
```

Three things to note.

**`config_path` is the ConfigMap mount.** `/app/configs/training_config.yaml`
is the volume, not the copy baked into the image — the ConfigMap is genuinely
in control.

**`data_seconds: 1.4`, against 1992.6 on the cold run.** The dataset PVC turned
a 33 minute download into a 1.4 second read. This is the measurement behind
D-024.

**Metrics are identical to the Docker run**, to four decimal places, on a
different runtime and a different filesystem hours apart. Same seed, same
`NUM_WORKERS=2`. This is the strongest available demonstration of D-021.

### Serving rollout

```
$ kubectl apply -f k8s/serving-deployment.yaml
$ kubectl apply -f k8s/serving-service.yaml
$ kubectl apply -f k8s/hpa.yaml

$ kubectl rollout status deployment/model-serving -n ml-training
deployment "model-serving" successfully rolled out
```

### Pods running and healthy

```
$ kubectl get pods -n ml-training -o wide
NAME                            READY   STATUS      RESTARTS   AGE     IP            NODE
model-serving-b6b78767f-8fgpf   1/1     Running     0          8m33s   10.244.0.11   minikube
model-serving-b6b78767f-f2th9   1/1     Running     0          8m48s   10.244.0.9    minikube
model-training-vjkwn            0/1     Completed   0          8m42s   10.244.0.10   minikube

$ kubectl get svc,hpa -n ml-training
NAME                    TYPE        CLUSTER-IP      PORT(S)   AGE
service/model-serving   ClusterIP   10.96.208.247   80/TCP    31m

NAME                                REFERENCE                  TARGETS                        MINPODS MAXPODS REPLICAS
horizontalpodautoscaler/model-serving  Deployment/model-serving  cpu: 0%/70%, memory: 43%/80%   2       5       2
```

Two replicas, both `1/1`, zero restarts. The Job pod is `Completed` — Jobs leave
their pod behind for log retrieval rather than deleting it.

### `describe deployment` — Part E requirements

```
$ kubectl describe deployment model-serving -n ml-training
Replicas:               2 desired | 2 updated | 2 total | 2 available | 0 unavailable
StrategyType:           RollingUpdate
RollingUpdateStrategy:  0 max unavailable, 1 max surge
  Containers:
   server:
    Image:      mlops-serve:v1
    Port:       8080/TCP (http)
    Limits:     cpu: 1        memory: 2Gi
    Requests:   cpu: 500m     memory: 1Gi
    Liveness:   http-get http://:http/health delay=0s timeout=1s period=10s #success=1 #failure=3
    Readiness:  http-get http://:http/health delay=15s timeout=1s period=5s #success=1 #failure=3
    Startup:    http-get http://:http/health delay=0s timeout=1s period=10s #success=1 #failure=270
    Mounts:     /app/checkpoints from checkpoints (ro)
```

Every Part E value is visible here: 2 replicas, rolling update 1/0, requests
500m/1Gi, limits 1/2Gi, liveness every 10s with failureThreshold 3, readiness
every 5s with a 15s delay, and the checkpoint volume mounted `(ro)`.

### Prediction through the Service

```
$ kubectl port-forward svc/model-serving 8080:80 -n ml-training

$ curl -s http://localhost:8080/health
{"status":"ok","model_loaded":true}

$ curl -s http://localhost:8080/metadata
{"class_names":["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"],
 "architecture":"resnet18","num_classes":10,
 "checkpoint_path":"/app/checkpoints/classifier_v1.pt",
 "trained_epochs":1,"val_loss":1.9177873344421388,"val_accuracy":0.296}

$ curl -s -X POST http://localhost:8080/predict -F "image=@test_image.png"
{"filename":"test_image.png","predicted_class":"cat","confidence":0.2227,
 "probabilities":{"airplane":0.0774,"automobile":0.0124,"bird":0.1779,"cat":0.2227,
 "deer":0.2091,"dog":0.1492,"frog":0.0588,"horse":0.0076,"ship":0.0724,"truck":0.0125}}
```

True label is `cat`, so the prediction is correct. `trained_epochs: 1` shows the
serving pods are running the weights early stopping preserved when epoch 2
regressed — the Job and the Deployment agree on which model won, having
communicated only through the shared PVC.

### The probe failure and its fix

The first cluster run exposed a genuine defect. Captured mid-run, both states
appear together:

```
NAME                             READY   STATUS             RESTARTS        AGE
model-serving-5bd87fbdd7-j4lqc   0/1     CrashLoopBackOff   7 (4m40s ago)   16m   <- no startupProbe
model-serving-5bd87fbdd7-xpmsp   0/1     CrashLoopBackOff   7 (4m40s ago)   16m   <- no startupProbe
model-serving-676cfc9865-8xlnp   0/1     Running            0               12m   <- startupProbe added
model-training-h9nnt             1/1     Running            0               36m
```

Identical images receiving identical 503 responses. The two without a
`startupProbe` had restarted seven times; the one with it had not restarted at
all. Full reasoning in D-023.

Throughout, the readiness probe was correct:

```
$ kubectl get endpoints model-serving -n ml-training
NAME            ENDPOINTS   AGE
model-serving               2m32s
```

Empty endpoints — the Service refused to route to pods that could not serve.

When the checkpoint finally appeared, every pod became ready without
intervention, and the stalled rollout completed:

```
model-serving-5bd87fbdd7-xpmsp   1/1   Running       8 (5m42s ago)   17m
model-serving-676cfc9865-8xlnp   1/1   Running       0               13m
model-serving-5bd87fbdd7-j4lqc   1/1   Terminating   8 (5m46s ago)   17m
```

Old pods terminated only *after* a new pod reported Ready, which is
`maxUnavailable: 0` doing its job.

### Part F requirement coverage

| Req | Claim | Evidence |
|---|---|---|
| F1 | Apply namespace, configmap, training job | All applied; Job reached `Complete 1/1` |
| F2 | Deploy serving layer and HPA | Rollout succeeded; HPA active with real targets |
| F3 | Pods running and healthy | 2/2 `1/1 Running`, 0 restarts; `describe deployment` captured |
| F4 | Port-forward and test `/predict` | 200 with a correct prediction and a valid distribution |
