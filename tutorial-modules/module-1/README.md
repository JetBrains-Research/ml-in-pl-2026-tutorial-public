# Module 1: IdeGYM sandboxes, from a laptop to a cluster

The ML in PL 2026 IdeGYM tutorial: what IdeGYM is, how to drive it, and how agents connect to
its sandboxes. Module 2 (NeMo Gym, Practical 2) lives in its own folder next to this one.

[All tutorial modules](../README.md) · [Repository overview](../../README.md) · [Module 2: NeMo Gym](../module-2/README.md)

This module is one [uv](https://docs.astral.sh/uv/) project. Each sketch is a plain script
that creates its own IdeGYM client and server, does one thing, and deletes the server on exit.
Every sketch reads one `.env`, so moving from the local cluster to the shared one means editing
only that file.

| Step | What we do | Sketch |
|---|---|---|
| 1 | [Install IdeGYM together on Minikube](#step-1-install-idegym-together) | `setup/install.sh` |
| 2 | [Live-code a hello world](#step-2-hello-world-live-coded) | `01_hello.py` |
| 3 | [Files and a small dataset](#step-3-files-and-a-dataset) | `02_files.py`, `03_dataset.py` |
| 4 | [Switch `.env` to the shared cluster and rerun steps 2–3](#step-4-the-same-sketches-on-the-shared-cluster) | — |
| 5 | [Build an image from plain Debian on the cluster](#step-5-build-an-image) | `04_build_image.py` |
| 6 | [Snapshot a server and start a second one from it](#step-6-snapshots) | `05_snapshots.py` |

## Before the workshop

You need macOS (Intel or Apple Silicon) or Linux with Docker running. Give Docker at least 4 CPUs
and 8 GB of RAM. You also need `minikube`, `kubectl`, `helm`, and `uv`. On macOS:

```bash
brew install minikube kubernetes-cli helm uv
```

Then, from `tutorial-modules/module-1/`:

```bash
setup/check-prerequisites.sh
uv sync                  # one environment for every step, with the IdeGYM 0.13.0 client
setup/prepull.sh         # pull images and the chart while the network is good
cp .env.example .env     # the local defaults work as-is
```

`prepull.sh` caches every image, and both the chart and the orchestrator use
`imagePullPolicy: IfNotPresent`, so nothing is downloaded again on site.

## Step 1: install IdeGYM together

```bash
setup/install.sh
kubectl port-forward svc/idegym 8000:80 -n idegym     # keep this terminal open
curl http://localhost:8000/health                     # {"status":"healthy"}
```

[`install.sh`](setup/install.sh) does three things:

1. Starts Minikube with 4 CPUs and 8 GB of memory.
2. Loads the pre-pulled images.
3. Installs the published chart.

Locally, sandboxes run with the default container runtime, not gVisor. That is less isolation
than on the shared cluster, but there is nothing to patch and nothing extra to download.

Every version is pinned in [`setup/versions.env`](setup/versions.env). The dashboard is at
<http://localhost:8000/>.

## Step 2: hello world, live-coded

[`01_hello.py`](01_hello.py) already has the imports, the connection parameters, and the image.
We type the `async with IdeGYMClient(...)` and `async with client.with_server(...)` blocks
together, following its TODOs. The finished version is
[`solutions/01_hello.py`](solutions/01_hello.py).

```bash
uv run 01_hello.py
```

Expected output: the pod's kernel and user, then `python in the sandbox says: 42`. While it
runs, watch the pod appear and disappear with `kubectl get pods -n idegym -w` or on the
dashboard.

## Step 3: files and a dataset

| Sketch | Scenario | Expected output |
|---|---|---|
| [`02_files.py`](02_files.py) | Create a file, edit lines, download, and upload bytes | `hello, world`, then `hello, ML in PL`, then `256 /tmp/blob.bin` |
| [`03_dataset.py`](03_dataset.py) | Score a small dataset of `solve(data)` tasks in one sandbox, as a verifier would | `3/3` for each of the five tasks |

For `03_dataset.py`, break one of the solutions and rerun. A real model mistake from the
preliminary runs: `str.isalnum()` in `palindrome`.

## Step 4: the same sketches on the shared cluster

In `.env`, replace `IDEGYM_ORCHESTRATOR_URL`, `IDEGYM_AUTH_USERNAME`, `IDEGYM_AUTH_PASSWORD`, and
`IDEGYM_NAMESPACE` with the values handed out at the workshop. Set `IDEGYM_CLIENT_NAME` to your
own name. Then rerun the same sketches, without changing any code:

```bash
uv run solutions/01_hello.py
uv run 02_files.py
uv run 03_dataset.py
```

Find your servers on the workshop dashboard by your client name. From here on, everything
runs on the shared cluster.

## Step 5: build an image

[`04_build_image.py`](04_build_image.py) describes an image in Python, starting from plain Debian:

- system packages;
- a user;
- the IdeGYM server;
- the IdeGYM repository as the project.

It prints the Dockerfile the plugins generate. Then the orchestrator builds the image in the
cluster and pushes it, and the sketch starts a sandbox from it. The build takes several minutes.

```bash
uv run 04_build_image.py
```

## Step 6: snapshots

[`05_snapshots.py`](05_snapshots.py) runs two servers in a row:

1. The first server writes a file and is snapshotted, memory and disk included, then deleted.
2. The second server starts from that snapshot instead of the image, so the file is already
   there.

Pod snapshots are GKE-only: they need gVisor, a non-E2 node, and exactly the same server
configuration for the restore.

```bash
uv run 05_snapshots.py
```

## Extras

These show other ways to reach a sandbox. They are not part of the main flow:

| Sketch | What it shows | Run |
|---|---|---|
| [`extras/bash_agent.py`](extras/bash_agent.py) | An agent with one `bash` tool, executed through IdeGYM's bash API | `uv run extras/bash_agent.py` |
| [`extras/openenv.py`](extras/openenv.py) | OpenEnv `EnvClient` over the orchestrator's WebSocket forwarding | `uv run extras/openenv.py` |
| [`extras/parallel.py`](extras/parallel.py) | Several sandboxes at once from one client, the way rollouts run in parallel | `uv run extras/parallel.py` |

`bash_agent` and `openenv` need `BASH_IMAGE` in `.env`.

## Dashboard

TODO: choose which pages to walk through. The dashboard is at the orchestrator's root URL, and
you log in with the same credentials. Filter servers by your `IDEGYM_CLIENT_NAME`, then open a
server's pod page for its logs and events.

## Teardown

```bash
setup/teardown.sh     # deletes the whole Minikube cluster, including PostgreSQL's volume
```

## Notes for instructors

- `05_snapshots.py` needs three things on the cluster:
  - pod snapshots enabled on the orchestrator (`podSnapshot.enabled`);
  - the gVisor runtime class;
  - an `n2` node pool.

  Check this before the session.
- `04_build_image.py` needs the orchestrator's image builder (Kaniko or Cloud Build) and a
  registry the cluster can pull from. Try one build before the session. With many participants,
  consider one shared build on screen instead.
