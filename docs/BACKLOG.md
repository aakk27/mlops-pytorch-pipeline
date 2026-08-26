# Known Gaps and Follow-up Work

What this project does **not** do, and what I would change before anyone relied
on it. Written down rather than left implicit, so the limitations are visible to
a reviewer and actionable by whoever picks it up next.

Priority reflects what would matter first in a real deployment, not what would
score marks.

---

## 1. Deviations from the assignment brief

Conscious choices, each with reasoning recorded. Listed here so they are easy to
find rather than buried in a decision log.

| # | Deviation | Why | Reference |
|---|---|---|---|
| 1 | Demonstration runs use `SUBSET_FRACTION=0.05, MAX_EPOCHS=2` instead of the brief's bare `docker run` (10 epochs, full dataset) | A full CPU run is ~3 hours; the mechanism under test is identical | D-005 |
| 2 | GPU bonus manifest written but never executed | No NVIDIA hardware; Apple Metal is unreachable from a Linux container | D-017 |
| 3 | AI assistance disclosed once in the README rather than in every commit message | Author preference; the brief's wording asks for commit messages | D-014 |
| 4 | `k8s/pvc.yaml` added beyond the brief's six-file list | The brief requires PVCs without specifying a file; separating claims from workloads is cleaner | D-024 |
| 5 | A `startupProbe` added alongside the two required probes | Without it the required liveness configuration restarts pods indefinitely | D-023 |

**The full 10-epoch run is the one worth closing.** Everything else is defensible
as-is; that one is only unclosed because of wall-clock.

---

## 2. Correctness and robustness

### 2.1 `serve.py` cannot distinguish a missing checkpoint from a corrupt one
**Priority: high · Effort: small**

`ModelStore.load()` catches bare `Exception` and returns `False`. A truncated or
corrupt checkpoint therefore looks exactly like "training hasn't finished yet":
the pod stays unready and reports `No checkpoint at ...`, which is a lie. With
the startupProbe's 45-minute budget, a corrupt file wastes 45 minutes before
anything visible happens.

Should distinguish: file absent → 503 "waiting"; file present but unloadable →
log the exception and report a distinct error. A checksum written alongside the
checkpoint would make this cheap.

### 2.2 Checkpoints are overwritten, never versioned
**Priority: high · Effort: medium**

Every run writes `classifier_v1.pt` to the same path. A second Job silently
destroys the previous model, and there is no way to roll back to a known-good
one. The `v1` in the filename is decorative.

Minimum fix: write `classifier-{timestamp}-{git-sha}.pt` and maintain `latest`
as a symlink or a small pointer file. Properly: a model registry.

### 2.3 `ReadWriteOnce` PVCs only work on a single node
**Priority: medium · Effort: large**

The Job writes the checkpoint and the serving pods read it, which works here
only because every pod lands on the same minikube node. On a multi-node cluster
the serving replicas would fail to mount a volume already attached elsewhere.

Real fix is object storage (S3/GCS/MinIO) with the serving pods pulling the
checkpoint at startup, or `ReadWriteMany` backed by NFS. Both are a different
architecture, not a config change.

### 2.4 The startupProbe budget is a guess for anything larger
**Priority: low · Effort: small**

45 minutes was sized against a measured 33-minute cold download plus ~9 minutes
of training. A larger dataset or a full 10-epoch run would exceed it, and the
failure mode (pod restarts, budget resets) is quiet.

Better: decouple readiness from training entirely — the serving Deployment
should not be applied until the Job completes, or should pull from a registry
where a model always exists.

### 2.5 No graceful shutdown handling
**Priority: low · Effort: small**

`uvicorn` handles SIGTERM, but there is no `preStop` hook or
`terminationGracePeriodSeconds` tuning, and no draining. During a rolling update
an in-flight prediction could be cut. Inference here is fast enough that it has
not bitten, but it is unhandled rather than handled.

---

## 3. Security

### 3.1 The training Job runs as root
**Priority: medium · Effort: medium**

minikube's default StorageClass provisions root-owned `hostPath` directories, so
running the trainer unprivileged means fighting volume ownership. The serving
container does run as UID 1001, since it only reads.

Fix is `fsGroup` in the pod securityContext plus an init container to chown, or
a StorageClass that honours `fsGroup`. Skipped to avoid a day-three yak shave.

### 3.2 `readOnlyRootFilesystem: false` on the serving container
**Priority: low · Effort: small**

Left writable because torch and uvicorn touch temp paths. Should be `true` with
`emptyDir` volumes mounted at `/tmp` and any cache directory.

### 3.3 The API has no authentication or rate limiting
**Priority: medium (deployment-dependent) · Effort: medium**

`POST /predict` is open to anything that can reach the Service. Fine behind a
`ClusterIP` in a private cluster, unacceptable the moment an Ingress appears.
No request size cap either — a large upload is read fully into memory.

### 3.4 The Secret template is not wired to any workload
**Priority: low · Effort: small**

`k8s/secret.example.yaml` demonstrates the pattern and `.gitignore` protects
filled copies, but nothing consumes it because this pipeline authenticates to
nothing. It is a demonstration, not a working credential path.

---

## 4. Operations

### 4.1 No image registry
**Priority: high · Effort: medium**

Images are built directly into minikube's Docker daemon with
`eval $(minikube docker-env)`. Nothing is pushed anywhere, so the manifests are
not portable to any other cluster and `imagePullPolicy: IfNotPresent` is doing
load-bearing work.

Fix: push to GHCR from CI, tag by commit SHA, reference by digest.

### 4.2 Image tags are mutable
**Priority: high · Effort: small**

Everything is `:v1`. Rebuilding changes what `:v1` means, so a manifest that
worked yesterday may deploy different code today with no diff to show for it.
Tag by SHA.

### 4.3 The HPA has never actually scaled
**Priority: medium · Effort: small**

It reports real metrics (`cpu: 0%/70%`) and is correctly wired, but no load test
has driven it past the threshold. So the manifest is verified; the *behaviour*
is not. A short `hey`/`ab` run against `/predict` would close this.

### 4.4 `ttlSecondsAfterFinished: 3600` deletes its own evidence
**Priority: low · Effort: trivial**

The Job self-deletes an hour after completing, taking its pod and logs with it.
Convenient for repeated demos, inconvenient when investigating a run from
yesterday. Logs should be shipped somewhere before this matters.

### 4.5 No log aggregation, metrics or tracing
**Priority: medium · Effort: large**

Training emits clean JSON Lines to stdout, which is the right shape for a
shipper — but nothing ships it. The serving app exposes no `/metrics`, so there
is no request rate, latency or error-rate visibility. Prometheus plus a
`ServiceMonitor` would be the obvious step.

### 4.6 No PodDisruptionBudget or anti-affinity
**Priority: low · Effort: trivial**

Both serving replicas can be evicted simultaneously, and on a real cluster both
could land on the same node — making `replicas: 2` decorative for availability
purposes.

### 4.7 metrics-server needs manual enabling
**Priority: low · Effort: trivial**

`minikube addons enable metrics-server` is a documented manual step. Without it
the HPA reports `<unknown>` indefinitely. It should be part of a setup script
rather than a line in a runbook someone has to read.

---

## 5. ML quality

### 5.1 The model is barely trained
**Priority: high (if the model matters) · Effort: small**

~30% validation accuracy from 2 epochs on 5% of CIFAR-10. Chance is 10%, so it
has learned something, but nothing anyone would deploy. A full 10-epoch run on
the complete dataset should reach 70-80%.

This was a deliberate trade for iteration speed and is not a defect in the
pipeline — but every prediction demo in this repository is made by a weak model,
and that is worth stating plainly.

### 5.2 No learning-rate schedule
**Priority: medium · Effort: trivial**

Fixed Adam at 0.001 for the whole run. A cosine schedule or `ReduceLROnPlateau`
is a few lines and typically worth several accuracy points on CIFAR-10.

### 5.3 Validation is used for early stopping *and* reported as the headline metric
**Priority: medium · Effort: small**

Selecting the checkpoint on validation loss and then quoting that same split's
accuracy as the result is mildly optimistic — the split has been used for model
selection. A held-out test split would give an honest number.

### 5.4 No evaluation beyond accuracy
**Priority: low · Effort: small**

No confusion matrix, no per-class metrics. With ten classes and a weak model,
per-class recall would say considerably more than a single scalar.

### 5.5 The dataset download is a single point of failure
**Priority: low · Effort: small**

`www.cs.toronto.edu` took 33 minutes and is the only source. An internal mirror
or a pre-populated volume would make cold starts predictable.

---

## 6. Testing

### 6.1 No test covers the environment-override logic
**Priority: high · Effort: small**

`apply_env_overrides()` is the mechanism every Docker and Kubernetes run depends
on, and its behaviour — including silently ignoring malformed values — is
verified only by a scratch script that was never committed. It should be a
proper parametrised test.

### 6.2 No test for early stopping
**Priority: medium · Effort: small**

The patience logic has been exercised repeatedly by real runs and works, but
nothing pins it. A test feeding a synthetic loss sequence would catch a
regression in the counter or the "save only on improvement" rule.

### 6.3 No integration test in CI
**Priority: medium · Effort: medium**

CI lints, unit tests, and builds both images — but never runs a container. A
smoke test that trains for one epoch on a tiny subset and then curls `/predict`
would have caught the probe bug far earlier than a manual cluster run did.

### 6.4 No manifest validation in CI
**Priority: medium · Effort: small**

The `k8s/` manifests are only validated by being applied by hand. `kubeconform`
or `kubectl apply --dry-run=server` in CI would catch schema errors before they
reach a cluster.

### 6.5 Coverage is unmeasured
**Priority: low · Effort: trivial**

22 tests pass; what fraction of `src/` they touch is unknown. `pytest-cov` with
a floor would make gaps visible.

---

## 7. Documentation

### 7.1 No API schema documentation beyond the generated one
**Priority: low · Effort: trivial**

FastAPI serves `/docs` automatically, which is good, but the response models are
plain dicts rather than Pydantic models — so the generated schema describes very
little. Typed response models would fix both at once.

### 7.2 The runbook assumes macOS and minikube throughout
**Priority: low · Effort: small**

Every command is written for this machine. A reader on Linux with kind would
need to translate, particularly around `docker-env` and the Docker Desktop
memory limits.

---

## Suggested order

If someone picked this up tomorrow:

1. **Registry and SHA tags** (4.1, 4.2) — everything else in ops depends on it
2. **Checkpoint versioning** (2.2) — currently one Job run away from data loss
3. **Env-override and early-stopping tests** (6.1, 6.2) — cheap, and they guard the most load-bearing logic
4. **The full training run** (5.1, deviation 1) — one overnight run closes the last real gap against the brief
5. **Integration test in CI** (6.3) — would have caught the bug that cost the most time here
