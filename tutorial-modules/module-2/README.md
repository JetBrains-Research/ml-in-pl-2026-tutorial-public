# Module 2: NeMo Gym evaluations with IDEGym

Run four evaluations with NeMo Gym and remote IDEGym sandboxes. Each run saves
its scores, logs, and model/tool activity in `results/`.

[Repository overview](../../README.md) · [All tutorial modules](../README.md)

| Experiment | Task | Teaching script |
| --- | --- | --- |
| [1: Python](experiments/module_1_python/README.md) | Generate and verify `solve(data)` | [workshop.py](experiments/module_1_python/workshop.py) |
| [2: direct bash](experiments/module_2_direct_bash/README.md) | Rename calculator functions through the IDEGym bash API | [workshop.py](experiments/module_2_direct_bash/workshop.py) |
| [3: OpenEnv](experiments/module_3_openenv_bash/README.md) | The same calculator task through `Tbench2Env` | [workshop.py](experiments/module_3_openenv_bash/workshop.py) |
| [4: PyCharm MCP](experiments/module_4_jetbrains_mcp/README.md) | Repair inspection problems with MCP tools | [workshop.py](experiments/module_4_jetbrains_mcp/workshop.py) |

## Setup

You need Docker (or `uv` for local Python), IDEGym workshop credentials, and an
OpenRouter API key. Sandboxes run remotely; no local GPU or Kubernetes
setup is needed.

From `tutorial-modules/module-2/`, copy the example if you do not already have
a `.env` file:

```bash
cp .env.example .env
```

Fill in the IDEGym username/password and `OPENROUTER_API_KEY`. Keep
`MODEL_PROVIDER=openrouter` as set in the example.

`POLICY_MODEL_NAME` selects the model for all experiments; `--model` overrides
it for one run. The default is Gemma 4 26B A4B. 

## Run with Docker

From `tutorial-modules/module-2/`:

```bash
export RUNNER_IMAGE=ghcr.io/jetbrains-research/ml-in-pl-2026-tutorial-public/module-2:latest
docker pull "$RUNNER_IMAGE"
mkdir -p results
docker run --rm --init -it --cpus=2 --memory=4g --shm-size=1g --user "$(id -u):$(id -g)" --env-file .env --mount "type=bind,src=$(pwd)/results,dst=/app/results" "$RUNNER_IMAGE" 1 run
```

Replace `1 run` with `2 run`, `3 run`, or `4 run`. Append `--help` to see an
experiment's options, or `--golden` for the reference repair in experiments 2–4.
The mounted `results/` directory keeps the output after the container exits.

Use the native Docker platform on Apple Silicon. Each container uses up to
2 CPUs and 4 GiB of memory. To run experiments concurrently, start a separate
container in each terminal; they can share the results directory.

On Windows, use PowerShell with Docker Desktop's Linux containers:

```powershell
$RunnerImage = "ghcr.io/jetbrains-research/ml-in-pl-2026-tutorial-public/module-2:latest"
docker pull $RunnerImage
New-Item -ItemType Directory -Force results | Out-Null
$ResultsPath = (Resolve-Path results).Path
docker run --rm --init -it --cpus=2 --memory=4g --shm-size=1g --env-file .env --mount "type=bind,src=$ResultsPath,dst=/app/results" $RunnerImage 1 run
```

## Run with local Python

From `tutorial-modules/module-2/`:

```bash
uv sync --locked --extra openenv
./run-experiment 1 run
./run-experiment 2 run
./run-experiment 3 run
./run-experiment 4 run
```

The launcher uses the project's locked `.venv`. Run local experiments one at a
time: experiments 1, 2, and 4 share a Gym port.

Useful options (also available with Docker):

| Option | Purpose |
| --- | --- |
| `--help` | List the experiment's options |
| `--model MODEL` | Override the model |
| `--golden` | Grade the reference repair without inference (2–4) |
| `--limit 5` | Run all five Python samples (1; default: one) |
| `--step-limit N` | Set the agent's turn limit (2–4; default: 20) |
| `--run-timeout SECONDS` | Set the overall deadline (default: 900 for 1, 1800 for 2–4) |
| `--output-dir PATH` | Save to a specific new directory |

## Output layout

Each run creates `results/<experiment>/<UTC timestamp>-model/` (or `-golden`).
Existing output directories are rejected to avoid overwriting earlier runs.

Start with the generated `README.md` for the score and links to the evidence.
The terminal shows preparation, model turns, tool completion, and grading.

| Files | Contents |
| --- | --- |
| `run.json`, `summary.json` | Run status and completed evaluation scores |
| `console.log`, `progress.jsonl`, `error.log` | Terminal output, progress, and errors when applicable |
| `config.yaml`, `inputs/` | Configuration and task input snapshots |
| `rollouts*.jsonl`, `rollouts_aggregate_metrics.json` | NeMo Gym responses, failures, dispatched inputs, and aggregate metrics |
| `model-calls/`, `logs/`, `gym.log`, `hydra/` | Model request/response captures and Gym diagnostics |
| `trajectory.jsonl`, `trace.md`, `final-files.json` | Tool activity and submitted sources (2–4) |

Experiments also save their sandbox and grading evidence. Golden runs save
`reference.log` instead of model-call captures and Gym metrics. Failed runs keep
partial output without a score; a completed evaluation can score zero.
Results are local and Git-ignored.

## Build after editing

From `tutorial-modules/module-2/`:

```bash
docker build -t ml-in-pl-module-2:local .
export RUNNER_IMAGE=ml-in-pl-module-2:local
```

Then use the Docker run command above. Source edits reuse the dependency layer;
dependency changes rebuild it. Remote sandbox images are configured in
[experiments/images.py](experiments/images.py) and each experiment's
`config.yaml`; update both and rebuild the runner to change them.

## Development checks

```bash
uv run --locked --extra openenv python -m pytest -q
uv run --locked --extra openenv ruff check experiments tests
uv run --locked --extra openenv ruff format --check experiments tests
```
