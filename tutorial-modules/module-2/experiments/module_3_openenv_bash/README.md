# Module 3: official OpenEnv bash

Open [workshop.py](workshop.py), then run from `tutorial-modules/module-2/`:

```bash
./run-experiment 3 run
```

This module uses the official
[tbench2_env](https://github.com/huggingface/OpenEnv/tree/8d3455ab91d28b46349b8bf01fc243d339cda260/envs/tbench2_env)
in `TB2_MODE=local`. Shell commands execute **inside the IDEGym server container**;
no E2B API key or Docker socket is required at runtime.
IDEGym provisions independent agent and grader containers with `server_kind=openenv`
on port 8000 and forwards
OpenEnv WebSockets. The default deployment uses the `IDEGYM_WORKSHOP_*`
credentials and URL from `.env`; the selected orchestrator is printed at startup.
Gemma uses LiteLLM by default, temperature 0.6, with up to 20 model turns.

## Server image

The experiment uses
`ghcr.io/jetbrains-research/ml-in-pl-2026-tutorial-public/openenv-server-tbench2-env:latest`,
fixed in `experiments/images.py` and this module's `config.yaml`.
This repository does not build the OpenEnv server image.

```bash
./run-experiment 3 run
./run-experiment 3 run --golden
```

The image only needs the official `tbench2_env` server and its execution
libraries. No calculator, prompt, task directory, or verifier belongs in it.
Use `TB2_MODE=local` (upstream's default) and a single server process with one
session per container. The runner starts the official root-based image inside
gVisor so TB2 can stage grading files at `/tests` and `/logs`. No special
`TB2_WITHHOLD_TESTS` setting is required: the harness keeps the verifier on the
client until the model finishes. Leave `TB2_TASK_WORKDIR` unset so
upstream reads the working directory from the runtime task metadata.

Before the model runs, the harness prepares both agent and grader concurrently.
In each server, it resets the official environment to `/tmp`,
uses an official `exec` action to stage the calculator and task files, then
resets to the new task at `/tmp/ml-in-pl-calculator/task`. The project lives at
`/tmp/ml-in-pl-calculator/project`. No verifier is sent during preparation;
the harness confirms grading files are absent before exposing tools to the
model.
After the agent closes, only submitted sources are transferred to the fresh
grader; the trusted verifier is sent there immediately before evaluation.
All these files disappear when IDEGym deletes the sandboxes.

## Follow the code

Read `evaluate()` for the episode flow, then `main()` for configuration and
reporting. The same four stages appear in Modules 1–4:

1. **`prepare_task()`** uses the official OpenEnv interface. It bootstraps with
   `client.reset(task_path="/tmp")`, stages calculator files and task metadata
   with `step(Tbench2Action(action_type="exec", command=runtime_setup()))`, then
   selects the task with a second `reset`. Both generic agent and grader servers
   start concurrently; each isolated client worker prepares its own task.
   Verifier files stay on the client throughout the model episode.
2. **`run_task()`** applies the reference repair or calls the shared bounded
   model loop. The model sees one `bash(command)` tool derived from the official
   `Tbench2Action` schema. Only the agent receives model actions; its official
   local backend executes them inside the IDEGym container. [client.py](client.py)
   adds only the WebSocket authentication header.
3. **`grade_task()`** runs in the waiting fresh OpenEnv server. The harness
   exports the expected source files and closes the agent, then transfers those
   sources and uploads `verifier_setup()` to the grader with official `exec`
   actions. `Tbench2Action(action_type="evaluate")` invokes TB2's own verifier
   execution and reward handling. The harness requires its five checks and
   official reward to agree. Infrastructure failures remain errors.
4. **`report_results()`** saves the summary and prints all checks, the official
   reward and tool-call counts. `sandboxes.json` records the two distinct
   sandbox IDs. Grader protocol/client logs are under `grader/`; agent logs
   remain at the top level of the result directory.

`evaluate()` visibly composes prepare -> run -> export -> close agent -> grade.
The connection context runs `prepare_task()` inside an OpenEnv worker process,
using the same locked Python environment as Gym. The `openenv` extra in the root
`pyproject.toml` pins the official runtime and Terminal-Bench environment to the
same upstream Git commit; the launcher installs both automatically. Context
managers close both clients and delete both IDEGym servers, including on errors
or cancellation.
No task or custom environment implementation is baked into the image.

## Read the output and try a change

```bash
./run-experiment 3 run --golden   # reference transformation, no inference
./run-experiment 3 run --step-limit 10
```

Output follows preparation → model/tool episode → checks, with a 30-second waiting
status and a configurable overall deadline (`--run-timeout`, default 1800 seconds).
Each model run chooses its own local Gym head port.

Compare `trajectory.jsonl` with `openenv-protocol.jsonl` to follow the model call,
official action, and observation. Inspect `evaluation.json` for verifier output,
five check results, and upstream reward; `environment-state.json` records the
last action. Change the prompt in `data/task.jsonl`, then explain how observations
change the agent's next turn. Changes to the fixture, prompt, or verifier take effect on the next run;
no image rebuild is needed.

## Build another OpenEnv environment

Browse the official [OpenEnv environments](https://github.com/huggingface/OpenEnv/tree/main/envs)
for existing environments and examples to use as a starting point for a new one.
Follow the chosen environment's README to build its server image and configure
its dependencies. Check its execution backend: some environments use external
services such as E2B even when the OpenEnv server itself runs in a container.
Publish the image to a registry that IDEGym can pull from, and provision it with
`server_kind=openenv`.

Another environment also needs its corresponding official client, action and
observation types, task setup, and grading logic; changing the image alone does
not adapt those interfaces.


## Saved artifacts

Each run creates `results/module_3_openenv_bash/<UTC datetime>-model/`
(or `-golden` for a reference run). `--output-dir` overrides this location.
The terminal prints the path and saves its transcript in `console.log`.

Open the generated run `README.md` for the score, completion status, and links
to all evidence. Model runs keep Gym's scored rollouts, aggregate metrics,
failure records, prepared datasets, component logs, and native model-call
captures. Module-specific evidence remains alongside them. Failed runs keep
`run.json`, `error.log`, logs, and partial outputs without claiming a score.

See [the output layout](../../README.md#output-layout) for the file catalog.
