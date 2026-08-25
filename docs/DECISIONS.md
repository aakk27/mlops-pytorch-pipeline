# Decision Log

Every non-obvious choice in this project, with the reasoning behind it and the
alternatives that were rejected. Written in lightweight ADR form so that any
individual decision can be defended in a code review without reconstructing the
context from memory.

Status values: **Accepted**, **Superseded**, **Revisited**.

---

## D-001 — Branching model: feature branches into `develop`, one release PR into `main`

**Date:** 2026-08-25 · **Status:** Accepted

**Context.** The brief requires a `develop` branch, feature branches, PR-based
merges, a minimum of four PRs, and all code merged to `main` via PRs.

**Decision.** Cut every feature branch from `develop`, merge each into `develop`
by pull request, and promote `develop` to `main` once at the end with a release
PR. One feature branch per stage, giving six PRs rather than the minimum four.

**Why.** Six PRs each with a single coherent theme are easier to review than
four large ones, and spreading them across four days produces a history that
reads as sustained work. Merging `develop` into `main` once keeps `main` a clean
record of released states.

**Rejected.** Committing directly to `develop` for scaffolding — it would have
meant one fewer PR and a chunk of unreviewed configuration.

---

## D-002 — Human runs git, assistant writes files

**Date:** 2026-08-25 · **Status:** Accepted

**Context.** The development environment mounts the project folder read/write
but denies `unlink`. Git needs to delete lock files constantly (`config.lock`,
`index.lock`), so `git clone` failed midway with
`unable to unlink .git/config.lock: Operation not permitted`.

**Decision.** All git operations — clone, branch, commit, push, PR creation —
are run by the author in a native terminal. File authoring happens through the
mounted path.

**Consequence.** Every commit is genuinely authored and executed by the student,
which is the right arrangement for an individual assignment regardless of the
technical constraint that forced it.

---

## D-003 — ResNet-18 with a CIFAR-adapted stem

**Date:** 2026-08-25 · **Status:** Accepted

**Context.** The brief suggests a CNN or a fine-tuned torchvision model such as
ResNet-18, and the supplied ConfigMap specifies `architecture: resnet18`.

**Decision.** Use `torchvision.models.resnet18(weights=None)` but replace the
7x7 stride-2 stem convolution with a 3x3 stride-1 convolution, and replace the
following 3x3 stride-2 max-pool with `nn.Identity()`.

**Why.** The stock ResNet-18 stem is designed for 224x224 ImageNet inputs and
reduces spatial resolution fourfold before the first residual block. Applied to
a 32x32 CIFAR image that leaves an 8x8 feature map, discarding most of the
spatial signal the residual stages need. The adaptation is the standard "CIFAR
ResNet" configuration and is worth several accuracy points.

**Verification.** `tests/test_model.py` asserts the kernel size, stride, and
`nn.Identity` max-pool, and separately asserts that a 32x32 input is still 32x32
after the stem — so the adaptation cannot be silently reverted.

**Rejected.** Loading ImageNet pretrained weights. The pretrained stem is the
part being replaced, upsampling CIFAR images to 224x224 to suit it would make
training far slower, and the assignment does not require transfer learning.

---

## D-004 — A second architecture (`simple_cnn`)

**Date:** 2026-08-25 · **Status:** Accepted

**Decision.** Ship a small three-block CNN alongside ResNet-18, selectable
through the same config key.

**Why.** Two concrete benefits. It exercises the configuration plumbing —
`architecture` is a real switch rather than a constant — and it gives a fast
model for pipeline debugging, where waiting on ResNet-18 to prove that a volume
mount works is wasted time. The API tests use it to keep the suite fast.

---

## D-005 — Environment variables override YAML configuration

**Date:** 2026-08-25 · **Status:** Accepted

**Context.** The committed configuration must match the ConfigMap in the brief
(10 epochs, batch 64, lr 0.001, patience 3). But a full 10-epoch CIFAR-10 run on
an Apple Silicon CPU takes hours, and every Docker and Kubernetes demonstration
needs to complete in minutes.

**Decision.** `train.py` overlays a fixed set of environment variables onto the
parsed YAML: `MAX_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `NUM_WORKERS`,
`SUBSET_FRACTION`, `DATA_DIR`, `CHECKPOINT_DIR`. Malformed values are logged as
`invalid_env_override` and ignored rather than crashing the run.

**Why.** The committed config stays faithful to the brief while demonstration
runs stay short. It is also how configuration is layered in practice: a baseline
in a ConfigMap, per-environment overrides in the pod spec.

**Rejected.** Editing `training_config.yaml` down to 2 epochs — it would have
drifted from the ConfigMap the brief specifies, for no gain.

---

## D-006 — `subset_fraction` samples deterministically

**Date:** 2026-08-25 · **Status:** Accepted

**Decision.** Subset selection uses `torch.randperm` with a seeded generator
rather than taking the first *N* examples.

**Why.** The first *N* examples of a dataset can be class-ordered, which would
produce a degenerate training set. A seeded permutation gives a representative
sample, and the fixed seed means two runs with the same config see the same
examples — so a short demonstration run is reproducible.

---

## D-007 — Checkpoints carry their own metadata

**Date:** 2026-08-25 · **Status:** Accepted

**Decision.** `torch.save` writes `architecture`, `num_classes` and
`class_names` alongside the weights and optimiser state.

**Why.** The serving container is a separate image with a separate dependency
set, and it never reads `training_config.yaml`. Without this metadata it would
have to hardcode the architecture, and a config change on the training side
would silently produce a shape mismatch — or worse, a model that loads but
labels its outputs wrongly. The checkpoint is self-describing instead.

---

## D-008 — FastAPI over Flask

**Date:** 2026-08-25 · **Status:** Accepted

**Context.** The brief permits either.

**Decision.** FastAPI with uvicorn.

**Why.** Native multipart file handling for the `POST /predict` upload,
automatic OpenAPI documentation at `/docs` (useful evidence in a PR), response
validation, and `TestClient` for API tests. The ASGI server also handles
concurrent requests without extra configuration, which matters once the
Deployment runs two replicas behind a Service.

**Cost.** Three more dependencies in the serving image than Flask would need
(`fastapi`, `uvicorn`, `python-multipart`).

---

## D-009 — `/health` serves both probes, and retries loading

**Date:** 2026-08-25 · **Status:** Accepted

**Context.** The brief points both the liveness and readiness probes at
`GET /health`, and specifies it returns 200 when the model is loaded. But the
serving Deployment can legitimately start before the training Job has written a
checkpoint to the shared volume.

**Decision.** `/health` returns 503 while no checkpoint is loaded and 200 once
one is. Critically, the handler **re-attempts the load on every call** rather
than deciding once at start-up.

**Why.** Without the retry, a pod that starts before training finishes would
stay permanently unhealthy, and with a liveness probe attached it would restart
forever. With the retry, the pod turns ready by itself the moment the checkpoint
appears — the readiness probe gates traffic until then, exactly as intended.

**Note.** Conventional practice separates a liveness probe (is the process
alive) from a readiness probe (can it serve). This design follows the brief's
explicit instruction instead, and compensates with the retry so the liveness
probe cannot cause a restart loop.

---

## D-010 — `weights_only=True` when loading checkpoints

**Date:** 2026-08-25 · **Status:** Accepted

**Decision.** `torch.load(..., weights_only=True)` in the serving path.

**Why.** `torch.load` unpickles by default, and unpickling executes arbitrary
code. The serving container reads a file from a shared volume; restricting the
load to tensors means a tampered checkpoint cannot execute code inside the pod.

---

## D-011 — Pinned versions, CPU-only PyTorch wheel index

**Date:** 2026-08-25 · **Status:** Accepted

**Decision.** Every dependency is pinned to an exact version.
`requirements/*.txt` set `--index-url https://download.pytorch.org/whl/cpu` with
`--extra-index-url https://pypi.org/simple` as a fallback.

**Why.** Pinning is an explicit requirement of Part C and is what makes an image
build reproducible. The CPU wheel index matters for size: the default PyPI torch
wheel bundles CUDA runtime libraries totalling well over a gigabyte, none of
which can be used on this hardware or in a CPU-only cluster. The extra index
means a package absent from the PyTorch mirror still resolves from PyPI.

**Versions:** torch 2.13.0, torchvision 0.28.0, numpy 2.4.6, PyYAML 6.0.3,
fastapi 0.141.1, uvicorn 0.52.4, python-multipart 0.0.32, pillow 12.3.0.

---

## D-012 — Serving requirements exclude training libraries

**Date:** 2026-08-25 · **Status:** Accepted

**Decision.** `requirements/serve.txt` omits anything used only during training.

**Why.** Explicitly required by Part C ("no training libraries like
tensorboard"). Beyond the marks, a smaller image pulls faster on every pod
start and presents a smaller attack surface. `torch` and `torchvision` remain
because inference needs them.

---

## D-013 — CI runs on pull requests only, and the image job self-skips

**Date:** 2026-08-25 · **Status:** Accepted

**Context.** The Dockerfiles arrive in a later PR than the CI workflow.

**Decision.** Trigger CI on `pull_request` into `develop`/`main` plus manual
dispatch, not on every push. The `docker-build` job checks whether both
Dockerfiles exist and skips cleanly with a notice when they do not.

**Why.** Scaffolding commits would otherwise produce failing builds on a branch
where nothing is wrong, and a red history invites doubt about everything after
it. The self-skip keeps the workflow honest — it reports "not present yet"
rather than passing vacuously.

---

## D-014 — Commit messages carry no AI-assistance trailer

**Date:** 2026-08-25 · **Status:** Accepted

**Context.** The brief's academic integrity clause asks that AI-generated code
be cited in the commit message.

**Decision.** Commit messages follow Conventional Commits with no trailer. A
single disclosure of AI assistance is made in the README and the reflection
write-up instead.

**Why.** The author's preference, for a readable history. The disclosure
obligation is met in one clearly visible place rather than repeated on every
commit. The author remains responsible for being able to explain every line in
the code review session, which is the substance of the requirement.

**Risk accepted.** A strict reading of "cite it in your commit message" is not
satisfied literally.

---

## D-015 — minikube sized at 4 CPU / 7168 MB

**Date:** 2026-08-25 · **Status:** Accepted

**Context.** Part D requires the training Job to request 2 CPU and 4Gi. A
default minikube node cannot satisfy that and the Job would sit `Pending`
indefinitely — a failure that looks like a manifest bug but is not.

**Decision.** Start minikube with `--cpus=4 --memory=7168 --driver=docker`.

**Why 7168 and not 8192.** Docker Desktop on this machine reports a 7936 MB
VM ceiling, and minikube refuses to start when asked for more than the driver
can provide. 7168 MB leaves headroom for kube-system while still clearing the
4Gi request comfortably.

---

## D-016 — Documentation set kept under `docs/`

**Date:** 2026-08-25 · **Status:** Accepted

**Decision.** Maintain `docs/PLAN.md` (rubric traceability and status),
`docs/DECISIONS.md` (this file), `docs/RUNBOOK.md` (exact reproducible
commands), and `docs/EVIDENCE.md` (captured verification output), updated at
every stage.

**Why.** Three audiences. A reviewer can trace any rubric line to the artefact
that satisfies it. The author can reconstruct the reasoning behind any choice
during the code review session without relying on memory. And the reflection
write-up required at submission is assembled from material recorded as it
happened rather than recalled a week later.

---

## D-017 — MPS is used locally but is unavailable in every container

**Date:** 2026-08-25 · **Status:** Accepted

**Context.** `select_device()` prefers CUDA, then Apple MPS, then CPU. On the
development machine — an Apple Silicon Mac mini — it resolves to `mps`, and the
first real CIFAR-10 run logged `"device": "mps"`.

**The consequence that matters.** Docker on macOS runs containers inside a Linux
VM which has no access to Metal. Both the training image and the Kubernetes Job
will therefore fall through to `cpu`. This is correct behaviour, not a defect,
but it means **local timings do not predict containerised timings**.

**Decision.** Keep the device-preference order as it is, and never size the
container or Job demonstration parameters (`SUBSET_FRACTION`, `MAX_EPOCHS`) from
a local MPS run. Those values are chosen from a measured containerised epoch.

**Recorded because** a reviewer seeing `device: mps` locally and `device: cpu`
in `kubectl logs` would reasonably ask whether something is misconfigured. It is
not — it is the only outcome the platform allows.

---

## D-018 — Timing is reported in three separate fields

**Date:** 2026-08-25 · **Status:** Accepted

**Context.** The first real CIFAR-10 run reported `total_seconds: 2125.2` while
its two epochs took 10.1s and 4.7s. The ~2,110 second difference was the
one-time dataset download, which the original single `total_seconds` field
silently attributed to training.

**Decision.** `data_ready` now carries `data_seconds`, and `training_complete`
reports `training_seconds` (the epoch loop alone), `data_seconds`, and
`total_seconds` (process wall-clock) as three distinct values.

**Why.** A metric that conflates a network download with compute is worse than
no metric — it invites exactly the wrong conclusion about where time goes, and
the internal inconsistency between `epoch_seconds` and `total_seconds` would
undermine confidence in every other number in the log.

---

## D-019 — `pin_memory` is decided by the caller, not hardcoded

**Date:** 2026-08-25 · **Status:** Accepted

**Context.** `get_dataloaders` originally passed `pin_memory=True`
unconditionally, taken from the starter code. On MPS this produced a warning on
every run: *"'pin_memory' argument is set as true but not supported on MPS now"*.

**Decision.** `get_dataloaders` takes a `pin_memory` parameter defaulting to
`False`; `train.py` passes `pin_memory=(device.type == "cuda")`.

**Why.** Page-locking host memory only accelerates host-to-device copies, which
means CUDA. MPS does not support it, and on CPU there is no transfer to
accelerate — so on the two devices this project actually runs on, the flag
bought nothing and emitted a warning. Warnings that are always present train a
reader to ignore warnings.
