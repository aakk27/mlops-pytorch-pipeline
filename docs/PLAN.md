# Project Plan and Rubric Traceability

**Repository:** `aakk27/mlops-pytorch-pipeline`
**Assignment:** Assignment 2 — Deploying PyTorch ML Workloads with Docker & Kubernetes
**Course:** DA5402W — MLOps & Infrastructure for Machine Learning, IIT Madras
**Started:** 2026-08-25
**Last updated:** 2026-08-30

This document is the single source of truth for what the assignment asks, where
each requirement is satisfied in this repository, and what state it is in. It is
updated at the end of every stage.

---

## 1. Rubric traceability

Each requirement from the assignment brief is mapped to the artefact that
satisfies it, so a reviewer can go straight from a mark to the code.

### Part A — Repository Setup (15 points)

| # | Requirement | Where satisfied | Status |
|---|---|---|---|
| A1 | Public GitHub repository named `mlops-pytorch-pipeline` | [github.com/aakk27/mlops-pytorch-pipeline](https://github.com/aakk27/mlops-pytorch-pipeline) | Done |
| A2 | Prescribed directory structure | `src/`, `configs/`, `docker/`, `k8s/`, `requirements/`, `tests/`, `.github/workflows/` | Done |
| A3 | `develop` branch cut from `main` | `develop` | Done |
| A4 | All work on feature branches | 9 feature/docs branches, no direct commits to `develop` or `main` | Done |
| A5 | Every feature branch merged via PR with a meaningful description | PRs #1-#15 | Done |
| A6 | Minimum 2 PRs week 1, 2 PRs week 2 | 15 PRs spread across the working days, four of them releases to `main` | Done |
| A7 | Conventional Commits | `chore`, `feat`, `fix`, `ci`, `docs` scopes throughout | Done |
| A8 | `.gitignore` | `.gitignore` — datasets, checkpoints, `.env`, secret manifests | Done |
| A9 | Secrets management | `.env.example`, `k8s/secret.example.yaml`, `*secret*.yaml` excluded by `.gitignore` | Done |

### Part B — PyTorch Model (10 points)

| # | Requirement | Where satisfied | Status |
|---|---|---|---|
| B1 | CNN or fine-tuned torchvision model for CIFAR-10 | `src/model.py` — ResNet-18 with a CIFAR-adapted stem, plus `simple_cnn` | Done |
| B2 | Dataset loading with transforms and DataLoaders | `src/dataset.py` | Done |
| B3 | Training reads hyperparameters from YAML | `src/train.py::resolve_config_path`, `load_config` | Done |
| B4 | Metrics logged to stdout as JSON lines | `src/train.py::log` | Done |
| B5 | Checkpoint saved to a configurable path | `src/train.py`, `output.checkpoint_dir` + `CHECKPOINT_DIR` | Done |
| B6 | Early stopping | `src/train.py`, `training.early_stopping_patience` | Done |
| B7 | Serving app with `POST /predict` | `src/serve.py::predict` | Done |
| B8 | Serving app with `GET /health` returning 200 when loaded | `src/serve.py::health` | Done |

### Part C — Docker Containerization (30 points)

| # | Requirement | Where satisfied | Status |
|---|---|---|---|
| C1 | Multi-stage `docker/Dockerfile.train` | `docker/Dockerfile.train` | Done |
| C2 | All dependency versions pinned | `requirements/train.txt`, `requirements/serve.txt` | Done |
| C3 | Training reads config from mounted volume or env var | `TRAINING_CONFIG_PATH`, ConfigMap mount at `/app/configs` | Done |
| C4 | Serving image based on a slim Python image | `docker/Dockerfile.serve` | Done |
| C5 | Inference dependencies only, no training libraries | `requirements/serve.txt` | Done |
| C6 | Exposes port 8080 | `docker/Dockerfile.serve` | Done — verified |
| C7 | Runs as a non-root user | `docker/Dockerfile.serve` | Done — `uid=1001(appuser)` |
| C8 | `HEALTHCHECK` instruction | `docker/Dockerfile.serve` | Done — container reports `(healthy)` |
| C9 | Local build and run verification evidence | `docs/EVIDENCE.md` Stage 2 | Done |

### Part D — Kubernetes Training Job (10 points, + 5 bonus)

| # | Requirement | Where satisfied | Status |
|---|---|---|---|
| D1 | `k8s/namespace.yaml` | `k8s/namespace.yaml` | Done — verified on cluster |
| D2 | `k8s/configmap.yaml` | `k8s/configmap.yaml` | Done — verified on cluster |
| D3 | Job uses the training image | `k8s/training-job.yaml` | Done — verified on cluster |
| D4 | ConfigMap mounted as a volume at `/app/configs` | `k8s/training-job.yaml` | Done — verified on cluster |
| D5 | PVCs for `/app/data` and `/app/checkpoints` | `k8s/pvc.yaml` | Done — verified on cluster |
| D6 | Resource requests and limits (2 CPU, 4Gi) | `k8s/training-job.yaml` | Done — verified on cluster |
| D7 | **Bonus:** GPU request with nodeSelector or toleration | `k8s/training-job.yaml` (documented, disabled) | Written, not executable on this hardware |

### Part E — Kubernetes Model Serving (5 points)

| # | Requirement | Where satisfied | Status |
|---|---|---|---|
| E1 | 2 replicas | `k8s/serving-deployment.yaml` | Done — verified on cluster |
| E2 | Checkpoint PVC mounted read-only at `/app/checkpoints` | `k8s/serving-deployment.yaml` | Done — verified on cluster |
| E3 | Liveness probe: `/health` every 10s, failureThreshold 3 | `k8s/serving-deployment.yaml` | Done — verified on cluster |
| E4 | Readiness probe: `/health` every 5s, initialDelay 15s | `k8s/serving-deployment.yaml` | Done — verified on cluster |
| E5 | Requests 500m/1Gi, limits 1/2Gi | `k8s/serving-deployment.yaml` | Done — verified on cluster |
| E6 | Rolling update, maxSurge 1, maxUnavailable 0 | `k8s/serving-deployment.yaml` | Done — verified on cluster |
| E7 | Service exposing port 80 to container port 8080 | `k8s/serving-service.yaml` | Done — verified on cluster |

### Part F — End-to-End Validation (10 points)

| # | Requirement | Where satisfied | Status |
|---|---|---|---|
| F1 | Apply namespace, configmap, training job | `docs/RUNBOOK.md` §5, `docs/EVIDENCE.md` | Done |
| F2 | Deploy serving layer and HPA | `docs/RUNBOOK.md` §5, `k8s/hpa.yaml` | Done |
| F3 | Verify pods running and healthy | `docs/EVIDENCE.md`, `docs/validation-screenshots.pdf` | Done |
| F4 | Port-forward and test `/predict` | `docs/EVIDENCE.md`, `docs/validation-screenshots.pdf` | Done |

### Submission

| # | Requirement | Where satisfied | Status |
|---|---|---|---|
| S1 | All code merged to `main` via PRs | Release PRs #7, #11, #13 and #15, `develop` → `main` | #7, #11, #13 merged, #15 pending |
| S2 | README with setup instructions | `README.md` | Done |
| S3 | README with architecture diagram | `README.md` (Mermaid) | Done |
| S4 | At least 4 merged PRs with meaningful descriptions | 15 PRs | Done |
| S5 | 300–500 word reflection write-up | `docs/REFLECTION.md` (489 words) | Done |

---

## 2. Point budget

| Part | Points | Notes |
|---|---:|---|
| A — Repository setup | 15 | |
| B — PyTorch model | 10 | |
| C — Docker | 30 | Heaviest single component |
| D — K8s training job | 10 | Under the brief's "Bonus: Kubernetes Deployment" heading |
| E — K8s serving | 5 | |
| F — End-to-end validation | 10 | |
| **Subtotal** | **80** | |
| GPU node bonus | 5 | Manifest only — no NVIDIA hardware on an Apple Silicon Mac |
| **Total available** | **85** | |

---

## 3. Stage plan

| Stage | Branch | PR | Scope | Status |
|---|---|---|---|---|
| 0 | `feature/project-scaffold` | #1 | Structure, `.gitignore`, `.dockerignore`, CI, `pyproject.toml`, `.env.example`, `Makefile` | **Merged** |
| 1 | `feature/pytorch-model` | #2 | `model.py`, `dataset.py`, `train.py`, `serve.py`, configs, pinned requirements, 22 tests | **Merged** |
| 1b | `docs/traceability` | #3 | `docs/PLAN.md`, `DECISIONS.md`, `RUNBOOK.md`, `EVIDENCE.md` | **Merged** |
| 2 | `feature/docker-training` | #4 | `Dockerfile.train`, `Dockerfile.serve`, local build/run evidence | **Merged** |
| 3 | `feature/k8s-deployment` | #5 | All `k8s/` manifests including HPA and the GPU bonus block | **Merged** |
| 4 | `feature/e2e-validation` | #6 | Cluster run evidence, README, architecture diagram, reflection | **Merged** |
| — | `develop` → `main` | #7 | First release PR — Parts A-F on `main` | **Merged** |
| 5 | `feature/full-training-run` | #8 | Full 10-epoch run on all of CIFAR-10, 87.18% validation accuracy | **Merged** |
| 6 | `feature/test-coverage` | #9 | 30 tests for config resolution, environment overrides and early stopping | **Merged** |
| 7 | `feature/ci-integration-test` | #10 | CI builds both images and runs the serving container end to end | **Merged** |
| — | `develop` → `main` | #11 | Second release PR — full run, tests, CI integration | **Merged** |
| 8 | `docs/final-updates` | #12 | Final plan and reflection updates | **Merged** |
| — | `develop` → `main` | #13 | Third release PR | **Merged** |
| 9 | `docs/validation-screenshots` | #14 | Submission screenshots and the 2026-08-30 re-run evidence | Open |
| — | `develop` → `main` | #15 | Fourth release PR | Pending |

Branching model: feature branches are cut from `develop` and merged into
`develop` by PR; `main` only ever receives a release PR from `develop`.

---

## 4. Schedule

| Day | Work | Notes |
|---|---|---|
| 1 | Stages 0, 1, 1b; install Docker, minikube, kubectl | Cluster verified `Ready` on day 1 — removes the largest schedule risk |
| 2 | Stage 2 — Docker images, local verification | Wall-clock dominated by the first image build |
| 3 | Stage 3 — Kubernetes manifests and cluster run | Highest-risk day |
| 4 | Stage 4 — README, diagram, reflection, first release PR | |
| 5-6 | Full training run, test coverage, CI integration test, final docs | Follow-up work after the deadline pressure was off |

One PR per day also produces a commit history that reads as sustained work
rather than a single burst, which matches the brief's "2 PRs per week" intent.

---

## 5. Open risks

| Risk | Impact | Mitigation | Status |
|---|---|---|---|
| Training Job stays `Pending` because the node cannot satisfy 2 CPU / 4Gi | Blocks Part D and F | minikube started with `--cpus=4 --memory=7168` | Mitigated |
| Docker Desktop VM capped below the requested minikube memory | Cluster will not start | Requested 7168MB against the 7936MB cap | Mitigated |
| minikube cannot pull locally built images | `ImagePullBackOff` | `eval $(minikube docker-env)` before building, `imagePullPolicy: IfNotPresent` | Resolved — no pull errors occurred |
| Serving pods cannot read the checkpoint the Job wrote | Part E fails | Single `ReadWriteOnce` PVC, single-node cluster, serving mounts it read-only | Resolved — verified end to end |
| HPA reports `<unknown>` targets | Part F evidence incomplete | metrics-server addon; resolves once pods are Ready | Resolved — `cpu: 0%/70%, memory: 43%/80%` |
| CIFAR-10 download slow or unavailable inside the Job | Training Job fails | Dataset PVC persists the download across runs; `subset_fraction` limits the work | Mitigated |
| GPU bonus cannot be executed | 5 bonus points | Manifest provided and documented as untested on this hardware; stated honestly | Accepted |
| Full 10-epoch CPU training exceeds the deadline | Schedule | `SUBSET_FRACTION` / `MAX_EPOCHS` env overrides for demonstration runs | Mitigated |
| No `linux/arm64` CPU wheel for the pinned torch | Docker build fails | PyPI fallback via `--extra-index-url` | Resolved — wheel installed in 65s |
| Containers are ~11x slower than local MPS | K8s Job demo too slow | Job sized from the measured containerised epoch (~52s), not the local one | Mitigated |


---

## 6. Late findings

Four things that only running the system revealed. The first two are recorded in
full in `docs/DECISIONS.md`:

**The liveness probe restart loop (D-023).** Serving pods entered
`CrashLoopBackOff` while waiting for a checkpoint that did not exist yet.
Corrected with a `startupProbe`. D-009's original reasoning is left in place and
marked as revisited rather than quietly rewritten.

**The dataset download cost (D-024).** The cold Job spent 1992.6 seconds
downloading CIFAR-10 against 520.5 seconds of training. With the PVC warm the
same step took 1.4 seconds. This is the measured justification for separating
the dataset and checkpoint claims.

**The CI job could not import `fastapi`.** The test job installed only
`requirements/train.txt`, so the serving tests were failing on a missing
dependency rather than on anything real. Fixed by installing both requirement
files; PR #10 then added an integration test that builds both images and drives
the serving container, so a break of that kind fails the build instead of
hiding in it.

**A stale `eval $(minikube docker-env)` produced three unrelated-looking
symptoms.** A 38-minute CIFAR-10 re-download, an empty `checkpoints-full/`, and
an out-of-memory kill all traced back to one shell still pointed at minikube's
Docker daemon. `eval $(minikube docker-env -u)` fixed all three at once.

---

## 7. Known gaps

`docs/BACKLOG.md` records what this project does not do: five conscious
deviations from the brief, and 28 items of follow-up work across correctness,
security, operations, ML quality, testing and documentation — each with a
priority, an effort estimate, and the reasoning behind leaving it.

Four of those items have since been closed across three PRs: the full 10-epoch run on all of
CIFAR-10 (PR #8, 87.18% validation accuracy), tests for the environment
overrides and for early stopping (PR #9), and the CI integration test (PR #10).
Nothing remains open against the brief itself — everything left in the backlog
is a documented trade rather than a missing requirement.
