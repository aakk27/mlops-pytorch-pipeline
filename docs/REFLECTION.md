# Reflection

Three things caught me out, and they all had the same shape: something worked at
one layer and broke at the next.

The first was a Kubernetes probe. My serving app returns 503 from `/health`
until a checkpoint loads, which is what lets the readiness probe keep unready
pods out of the Service. But the assignment points the *liveness* probe at the
same endpoint. In Docker that never mattered: I always started serving after
training finished. On the cluster the Deployment comes up while the Job is still
running. Liveness got three 503s, restarted the container, and did it again a
minute later. I watched pods reach four restarts before I understood what I was
looking at. The fix was a `startupProbe`, which suspends liveness until it
passes once — obvious afterwards. What bothers me is that my decision log
already explained why padding `initialDelaySeconds` was enough. It buys about
fifty seconds. The gap was forty minutes.

The second was hardware I couldn't use. The Mac mini has an Apple GPU and
PyTorch picks it up as MPS: 4.7 seconds an epoch on a subset. The same image
inside Docker reports `device: cpu` and takes 52. Metal is macOS-only, Docker
containers run in a Linux VM, and Kubernetes sits one layer deeper again. The
gap is architectural, and the GPU bonus manifest I wrote can be correct without
ever being executable here.

The third was CI. My tests passed locally and failed on the first clean
checkout: `ModuleNotFoundError: No module named 'fastapi'`. My virtualenv had
both requirements files installed, so CI was testing a configuration I had never
run. Thirty seconds to fix, and it taught me more than the fix suggests.

The problem that cost the most time wasn't in the code at all. My shell still
had `eval $(minikube docker-env)` set from the Kubernetes work, so every
`docker run` talked to minikube's daemon, not Docker Desktop's. The volume
mounts resolved inside the minikube node where those paths don't exist, and
Docker silently created empty directories. That produced a 38-minute
re-download, a checkpoint on the wrong filesystem, and an OOM kill inside a
container already capped for Kubernetes. Three symptoms, one stale environment
variable, and no error until the daemon went away.

Underneath all of it is the same lesson. Each layer — virtualenv, container,
orchestrator, CI runner — relies on assumptions the layer below happened to
satisfy. Containerisation is sold as "it runs the same everywhere". That is true
of the code. It is not true of the surroundings: what hardware you can reach,
what is already on disk, which daemon you are talking to, and when your
dependencies exist relative to when you need them.

If I did it again I would deploy to the cluster on day one, with a stub model
that returns nonsense. Every one of these appeared at an integration boundary,
and I did not reach one until day two.
