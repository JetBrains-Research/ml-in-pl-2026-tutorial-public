# Module 2: direct bash access

[All experiment modules](../../README.md) · [Run artifacts](../../README.md#output-layout)

## Start here

Open [workshop.py](workshop.py), then run from `tutorial-modules/module-2/`:

```bash
./run-experiment 2 run
```

Learn how a **multi-step shell agent** connects NeMo Gym to IDEGym. The central
script owns the real episode, SDK command execution and run orchestration.
The default model comes from `.env`: Gemma through LiteLLM. Use the configured
generic Bookworm image; task setup happens at runtime.

## Follow the code

Read `evaluate()` for the episode flow, then `main()` / `run_workshop()` for
configuration and reporting. The same four stages appear in Modules 1–4:

1. **`prepare_task()`** stages calculator files, installs pytest, and verifies
   the unmodified baseline. Agent and fresh grader sandboxes start concurrently,
   and both are prepared concurrently before the model runs. The generic
   Bookworm image contains no exercise files.
2. **`run_task()`** applies the reference repair or calls the shared bounded
   `run_episode()` loop. Only the agent exposes `bash` to the model.
   **`execute_direct()`** is the actual SDK boundary:
   `IdeGYMServer.execute_bash()`. It sets the task working directory and PATH;
   stdout, stderr and exit code become observations for the next model turn.
3. **`grade_task()`** receives only the expected source files exported from
   the agent. The agent is closed first. Submitted sources replace the grader's
   fixture; no runtime packages or processes cross the boundary. The five
   checks cover rename, references, arithmetic, division by zero and main output.
   All must pass, and a model run must have a successful tool call, for reward 1.
4. **`report_results()`** validates the rollout, saves `summary.json`, and prints
   checks and reward. `sandboxes.json` records distinct agent and grader IDs.

`evaluate()` visibly composes prepare -> run -> export -> close agent -> grade.
The shared `concurrent_sandboxes()` helper handles startup and cleanup; the
module keeps its SDK operations visible. Model and reference runs use the same
preparation and grading path. The Gym adapter delegates to this same episode.
Module 3 shares the calculator fixture, bounded model loop and five checks,
while using the official OpenEnv interface.

## Read the output and try a change

Preparation, episode and grading are labelled `[1/3]`, `[2/3]` and `[3/3]`.
Model runs show sandbox readiness, model turns and tool completion; raw command
arguments and responses stay in `trajectory.jsonl`. A waiting status appears
every 30 seconds. `--run-timeout` bounds the whole evaluation (default 1,800
seconds), while command timeouts and `--step-limit` bound different operations.
`--golden` checks the reference transformation without inference.

Try `--step-limit 2` and inspect `trajectory.jsonl` and `final-files.json`.
Explain how the model discovered the files, what it changed, and which check
failed if it ran out of turns. Compare the same run with Module 3: the tool
schema and grade stay the same, but command delivery changes.

Direct, non-MCP bash over IdeGYM’s authenticated HTTP forwarding. The model calls `bash`, and the harness executes `IdeGYMServer.execute_bash`.

Module 2 uses the calculator task and five independent checks. It exposes only direct bash, with a 20-turn agent loop. Module 3 uses the official OpenEnv `tbench2_env` in local mode: the same model-visible bash interface maps to `Tbench2Action` execution, followed by the environment’s evaluation action. No training updates occur.

```bash
./run-experiment 2 run --golden
./run-experiment 2 run --model vertex_ai/google/gemma-4-26b-a4b-it-maas
```

Module 2 uses the same generic image as Module 1:
`ghcr.io/jetbrains-research/ml-in-pl-2026-tutorial-public/idegym-server-20250520-slim:1.0.0`.
The image is fixed in `experiments/images.py` and this module's `config.yaml`.
The runner writes the
shared files from `data/calculator.json` into `/home/appuser/work`, installs
`pytest==8.4.2` at runtime, and validates the unmodified baseline before the model
acts. Projects and verifier commands are supplied by the runner; no calculator
project is baked into this image and no Module 2 image build is needed.


## Saved artifacts

Each run creates `results/module_2_direct_bash/<UTC datetime>-model/`
(or `-golden` for a reference run). `--output-dir` overrides this location.
The terminal prints the path and saves its transcript in `console.log`.

Open the generated run `README.md` for the score, completion status, and links
to all evidence. Model runs keep Gym's scored rollouts, aggregate metrics,
failure records, prepared datasets, component logs, and native model-call
captures. Module-specific evidence remains alongside them. Failed runs keep
`run.json`, `error.log`, logs, and partial outputs without claiming a score.

See [the output layout](../../README.md#output-layout) for the file catalog.
