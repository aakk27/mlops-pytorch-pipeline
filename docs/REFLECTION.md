# Reflection

Three things caught me out on this assignment, and all three had the same shape:
something worked perfectly at one layer and broke at the next.

The first was a Kubernetes probe. My serving app returns 503 from `/health`
until a checkpoint is loaded, which seemed sensible — the readiness probe then
keeps unready pods out of the Service. But the assignment points the *liveness*
probe at the same endpoint. In Docker that never mattered, because I always
started the serving container after training had already finished, so `/health`
returned 200 straight away. On the cluster the Deployment legitimately comes up
while the training Job is still running. Liveness got three 503s, restarted the
container, and then did it again about once a minute. I watched pods accumulate
four restarts before I worked out what I was looking at. The fix was a
`startupProbe`, which suspends liveness and readiness until it passes once —
obvious in hindsight. What bothers me is that I had already written a paragraph
in my decision log explaining why padding `initialDelaySeconds` was sufficient.
It wasn't, and no amount of local testing would have told me.

The second was hardware I couldn't use. The Mac mini has an Apple Silicon GPU
and PyTorch picks it up as MPS, so an epoch on 5% of CIFAR-10 takes 4.7 seconds.
The same image inside Docker reports `device: cpu` and takes 52. Metal is a
macOS framework, Docker containers on macOS run inside a Linux VM, and
Kubernetes sits one layer further in again. There is no flag that fixes this. I
spent a while looking for one before accepting that the 11x gap is
architectural, and that the GPU bonus manifest I wrote can be correct without
ever being executable on this machine.

The third was CI. My tests passed locally and then failed on the first clean
checkout with `ModuleNotFoundError: No module named 'fastapi'`. My virtualenv
had both requirements files installed, so a CI job installing only the training
set was exercising a configuration I had never actually run. That took thirty
seconds to fix and taught me more than the fix suggests: my own machine had been
quietly hiding a genuine gap in the dependency declarations.

Underneath all three is the same lesson. Each layer — virtualenv, container,
orchestrator, CI runner — makes assumptions that the layer below happened to
satisfy by accident. Containerisation gets sold as "it runs the same
everywhere", and that is true of the code. It is not true of the surroundings:
what hardware you can reach, what is already sitting on disk, and when your
dependencies exist relative to when you start needing them. Almost all of my
debugging time went there rather than into PyTorch.

If I did it again I would deploy to the cluster much earlier, even with a stub
model that returns nonsense. Every one of these problems appeared at an
integration boundary, and I did not reach a single one of those boundaries until
day two.
