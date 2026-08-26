# Runbook

Every command needed to reproduce this project from a clean machine, in order,
with the failures that were actually hit and how they were resolved.

Paths assume the repository root. Where a command must run elsewhere, it says so.

---

## 1. Environment

Verified on:

| Component | Version |
|---|---|
| Hardware | Apple Silicon Mac mini, 24 GB RAM |
| OS | macOS (Darwin 26.5.2, arm64) |
| Docker | Docker Desktop, engine 29.2.1 |
| minikube | 1.38.1 |
| kubectl | 1.36.4 |
| Kubernetes | v1.35.1 (minikube) |
| Python | 3.11 |

> **Paths contain spaces on this machine.** The project lives under
> `.../MTECH IITM/DA5402W MLOPS/PyTorch ML Workloads with Docker & Kubernet/`.
> Every shell argument interpolating `$(pwd)` must be quoted or it word-splits.
> This caused a real failure once — see §7.

---

## 2. One-time setup

```bash
# Toolchain
brew install gh minikube kubectl

# GitHub authentication. The `workflow` scope is required to push any commit
# that touches .github/workflows/, and is NOT included in the default scopes.
gh auth login                                   # GitHub.com -> HTTPS -> browser
gh auth setup-git                               # let git use the gh credentials
gh auth refresh -h github.com -s workflow

# Cluster. Docker Desktop must be running first.
minikube start --cpus=4 --memory=7168 --driver=docker
minikube addons enable metrics-server           # required for the HPA
kubectl get nodes                               # expect: minikube  Ready
```

`--memory=7168` rather than 8192: Docker Desktop on this machine caps its VM at
7936 MB and minikube refuses to start when asked for more than the driver can
supply. See `docs/DECISIONS.md` D-015.

---

## 3. Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/train.txt -r requirements/serve.txt
pip install pytest==9.1.1 httpx==0.28.1 ruff==0.16.4
```

### Lint and test

```bash
make lint          # ruff check + ruff format --check
make test          # pytest tests -v
```

### A short training run

The committed `configs/training_config.yaml` matches the ConfigMap in the brief
(10 epochs, full dataset). Environment variables shorten a run without editing
it:

```bash
TRAINING_CONFIG_PATH=configs/training_config.yaml \
DATA_DIR=./data \
CHECKPOINT_DIR=./checkpoints \
SUBSET_FRACTION=0.05 \
MAX_EPOCHS=2 \
NUM_WORKERS=0 \
python src/train.py
```

The first run downloads CIFAR-10 (~170 MB) into `DATA_DIR`; later runs reuse it.

Available overrides: `MAX_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `NUM_WORKERS`,
`SUBSET_FRACTION`, `DATA_DIR`, `CHECKPOINT_DIR`, `TRAINING_CONFIG_PATH`.

### Serving locally

```bash
MODEL_CHECKPOINT_PATH=./checkpoints/classifier_v1.pt python src/serve.py

curl -s localhost:8080/health | jq
curl -s localhost:8080/metadata | jq
curl -s -X POST localhost:8080/predict -F "image=@test_image.png" | jq
```

Interactive API documentation is served at <http://localhost:8080/docs>.

### Generating a test image

`test_image.png` is git-ignored, so create one locally:

```bash
python - <<'PY'
from torchvision import datasets
ds = datasets.CIFAR10(root="./data", train=False, download=True)
img, label = ds[0]
img.resize((128, 128)).save("test_image.png")
print("saved test_image.png, true label:", ds.classes[label])
PY
```

---

## 4. Docker

### Build

```bash
make build                # both images
# or explicitly:
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
```

First build takes 10-20 minutes (torch download); rebuilds after a code-only
change take seconds, because requirements are installed in an earlier layer.

### Verify the image properties

```bash
docker images | grep mlops
docker run --rm mlops-serve:v1 id                                          # non-root
docker inspect --format='{{json .Config.Healthcheck}}'  mlops-serve:v1
docker inspect --format='{{json .Config.ExposedPorts}}' mlops-serve:v1
docker run --rm mlops-serve:v1 ls -la /app/src                             # inference modules only
```

### Run training in a container

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  -e SUBSET_FRACTION=0.05 -e MAX_EPOCHS=2 -e NUM_WORKERS=2 \
  mlops-train:v1
```

Expect `"device": "cpu"` — containers on macOS have no Metal access (D-017), so
this is roughly 11x slower per epoch than the local MPS run. Around 105 seconds
for the parameters above.

### Run serving in a container

```bash
docker run --rm -p 8080:8080 -v "$(pwd)/checkpoints:/app/checkpoints" mlops-serve:v1
```

In another shell:

```bash
curl -s localhost:8080/health   | jq
curl -s localhost:8080/metadata | jq
docker ps --format '{{.Image}}\t{{.Status}}'     # expect (healthy) after ~35s
curl -s -X POST localhost:8080/predict -F "image=@test_image.png" | jq
```

---

## 5. Kubernetes

*(Stage 3 — expanded once the manifests land.)*

Build the images **inside** the minikube Docker daemon so the cluster can find
them without a registry. This must be done in the same shell that runs the
builds:

```bash
eval $(minikube docker-env)
make build
docker images | grep mlops        # confirm they exist in minikube's daemon
```

Manifests set `imagePullPolicy: IfNotPresent` so the kubelet never tries to pull
from Docker Hub.

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/training-job.yaml

kubectl get pods -n ml-training -w
kubectl logs -f job/model-training -n ml-training

# Once training completes
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml

kubectl get pods -n ml-training
kubectl describe deployment model-serving -n ml-training

kubectl port-forward svc/model-serving 8080:80 -n ml-training
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

Teardown:

```bash
make k8s-down      # kubectl delete namespace ml-training
minikube stop
```

---

## 6. Git workflow

```bash
git checkout develop && git pull
git checkout -b feature/<name>

# ... changes ...

git add <paths>
git status --short                 # always review before committing
git commit -m "type(scope): summary" -m "Body explaining why."
git push -u origin feature/<name>

gh pr create --base develop --head feature/<name> \
  --title "type: summary" --body-file "../pr-body-<name>.md"
```

PR body files are kept **outside** the repository so they are never committed.

Commit types in use: `feat`, `fix`, `chore`, `ci`, `docs`, `test`.

---

## 7. Troubleshooting

Failures actually encountered during this project, and their resolutions.

| Symptom | Cause | Fix |
|---|---|---|
| `Exiting due to MK_USAGE: Docker Desktop has only 7936MB memory but you specified 8192MB` | minikube asked for more memory than the Docker Desktop VM has | Lower to `--memory=7168`, or raise Docker Desktop's limit in Settings → Resources |
| `The connection to the server localhost:8080 was refused` from `kubectl` | No cluster running yet | Start minikube first; harmless if it appears before `minikube start` succeeded |
| `remote: Invalid username or token. Password authentication is not supported` | GitHub removed password auth for git over HTTPS | `gh auth login` then `gh auth setup-git` |
| `refusing to allow an OAuth App to create or update workflow .github/workflows/ci.yml without workflow scope` | Token lacks the `workflow` scope | `gh auth refresh -h github.com -s workflow` |
| `pull request create failed: No commits between develop and feature/x` | The commit never ran — a multi-line heredoc pasted into the shell was swallowed | Verify with `git log --oneline -2`; use `git commit -m ... -m ...` instead of a heredoc |
| `docker: invalid reference format: repository name (DA5402W) must be lowercase` | Unquoted `$(pwd)` word-split on spaces in the project path, so Docker read a path fragment as the image name | Quote every volume argument: `-v "$(pwd)/data:/app/data"`. The Makefile quotes `$(PWD)` for the same reason (D-022) |
| `ModuleNotFoundError: No module named 'torchvision'` in a working shell | New terminal, virtualenv not active | `source .venv/bin/activate` |
| `unable to unlink .git/config.lock: Operation not permitted` | Git run against a mount that denies deletes | Run all git commands from a native terminal (D-002) |
| Training Job stuck `Pending`, `describe` shows `Insufficient cpu/memory` | Node smaller than the Job's 2 CPU / 4Gi request | Restart minikube with `--cpus=4 --memory=7168` |
| Pod in `ImagePullBackOff` for `mlops-train:v1` | Image built in the host Docker daemon, not minikube's | `eval $(minikube docker-env)` then rebuild; keep `imagePullPolicy: IfNotPresent` |
| Serving pods never become ready | No checkpoint on the shared volume yet | Expected until the Job finishes — `/health` retries the load and the pod becomes ready on its own. Confirm the Job succeeded and that both mount the same PVC |
| `kubectl top` / HPA shows `<unknown>` | metrics-server not installed | `minikube addons enable metrics-server`, then wait ~60s |
| No output from `kubectl logs` during training | Buffered stdout in a non-TTY | Already handled: `print(..., flush=True)` and `PYTHONUNBUFFERED=1` in the image |

---

## 8. Verification checklist

Run before opening the final release PR.

- [ ] `make lint` clean
- [ ] `make test` — all tests pass
- [ ] Both images build from a clean context
- [ ] Serving image runs as a non-root user (`docker run --rm mlops-serve:v1 id`)
- [ ] `docker inspect` shows a `Healthcheck` on the serving image
- [ ] Training Job reaches `Completed` and wrote a checkpoint to the PVC
- [ ] Both serving replicas `Running` and `1/1 Ready`
- [ ] `curl /predict` through the port-forward returns a class and probabilities
- [ ] HPA reports a real target rather than `<unknown>`
- [ ] README renders, architecture diagram included
- [ ] All feature PRs merged into `develop`, release PR opened into `main`
