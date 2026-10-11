# Module 1: generate and verify a Python function

[All experiment modules](../../README.md) · [Run artifacts](../../README.md#output-layout)

Open **[workshop.py](workshop.py)**. This is the central script for the exercise:
its sample loading, Gym service configuration, verifier, and evaluation entry point show
how NeMo Gym and IDEGym work together.

All prompts and test cases live in **[data/sample.jsonl](data/sample.jsonl)**,
one sample per line. The default command evaluates the first sample (factorial).
The model writes a `solve(data)` function in one response. NeMo Gym passes that
response to our verifier. The verifier uploads the generated source to a fresh
IDEGym sandbox, executes three test cases, compares the outputs, and returns a
reward. The model has no tools and does not receive test feedback to repair its
answer. No model weights are downloaded and no parameter updates occur.

```text
Prompt → NeMo Gym SimpleAgent → model through LiteLLM
                 ↓ generated Python
          our verifier → IDEGym sandbox → test outputs → reward
```

## Start the exercise

From the repository root:

```bash
cd tutorial-modules/module-2
./run-experiment 1 run
```

Install `uv` and fill in `.env` using `.env.example` before running. The launcher
installs Python 3.13.14 and the locked project dependencies automatically. Gym
comes from the pinned Git commit in `pyproject.toml`; you do not need a local Gym
checkout, Docker, a GPU, or Kubernetes credentials. IDEGym provisions the remote
sandbox. Gym also starts a local CPU Ray runtime.

The shared setup currently selects Gemma 4 26B A4B through LiteLLM and the
original, non-workshop IDEGym endpoint. Credentials stay in `.env`; exported
settings take precedence. The sandbox uses
`ghcr.io/jetbrains-research/ml-in-pl-2026-tutorial-public/idegym-server-20250520-slim:1.0.0`,
the same generic Bookworm runtime as Module 2. This module needs no image build.

You can run the central file directly with the same dependencies:

```bash
uv run --locked --extra openenv python experiments/module_1_python/workshop.py
```

## Follow the code

Read `evaluate()` for the episode flow, then `main()` for configuration
and reporting. The same four stages appear in Modules 1–4:

1. **`prepare_task()`** describes the generic image and runtime upload using
   NeMo Gym's `SandboxSpec`. Only generated `solution.py` is uploaded; expected
   answers stay on the client. Each answer gets a fresh grading sandbox.
2. **`run_task()`** executes the generated answer for each input through
   `AsyncSandbox.exec()`. Gym's `SimpleAgent` has already generated the answer
   with `max_steps=1`; this stage has no model tools or repair loop.
3. **`grade_task()`** compares the captured outputs with expected JSON values
   and types on the client. Every case must pass for reward 1. Execution errors
   and timeouts fail checks; infrastructure failures abort evaluation.
4. **`report_results()`** shows generated code, individual cases and rewards,
   then saves `summary.json`. No model parameters are updated.

`evaluate()` composes preparation, execution and grading and owns the
sandbox context. `main()` loads the samples with `load_samples()`, snapshots
`gym_configuration()`, launches `gym eval run`, and reports its scored rollouts.
The configuration wires Gym's inference service, `SimpleAgent`, and resources
server. Expected answers in `verifier_metadata` are kept out of the model prompt.

The server adapter in `resources_servers/idegym_python/app.py` delegates to this
same flow through its `verify_solution()` callback. `workshop_support.py` contains parsing, result formatting, and
process plumbing. Edit the sample file to change prompts and test cases; edit
the central script to change the configuration and verification logic.

## Read the output

The terminal identifies the task, model, and remote execution, then reports:

```text
[1/3] Preparing samples and NeMo Gym evaluation services...
[2/3] NeMo Gym is generating and grading 1 solution(s)...
[2/3] factorial: Model response received. Starting an IDEGym sandbox...
[2/3] factorial: Sandbox ready. Executing 3 test cases...
[2/3] factorial: Test execution finished; sandbox closed.
[3/3] Evaluation complete; Gym services stopped.
```

It then shows the generated source, each test's actual and expected output,
and the reward. These messages follow actual verifier progress; raw Ray and
HTTP logs go to files rather than filling the terminal.

The artifact/log directory is printed before Gym starts. While waiting, the runner
prints a status message every 30 seconds. The first verifier event means the model
has responded; before that, it may be waiting on service startup or inference.
The whole Gym run has a 900-second deadline, including all selected samples and
repeats. Change it with `--run-timeout 1800` for a longer batch. A deadline stops
local Gym processes, preserves logs/partial results, and reports an incomplete
evaluation without assigning a score. `--test-timeout` controls each sandbox test
separately. Ctrl+C also stops the local run; a remote sandbox already being graded
may need IDEGym cleanup if the service is interrupted before its normal teardown.

Reward is **1 only when every case passes**, otherwise **0**. Wrong answers,
syntax/runtime errors, empty answers, and per-test timeouts score 0. A failure
to access the model or sandbox prevents the evaluation from finishing and is
reported as an error, not a zero score. Tests compare JSON types as well as
values, so `true` is not accepted as the integer `1`.

## Try changing it

Run again after each change and inspect the generated code and results:

1. Add `{"input": 1, "expected": 1}` to the factorial row's `test_cases` in
   `data/sample.jsonl`. Keep each sample on one JSONL line. Confirm the terminal
   now shows four cases and the saved task snapshot includes the new case.
2. Change the user prompt to ask for a recursive implementation. Does the
   returned code change? Do the same tests still pass?
3. Change one expected answer deliberately. What happens to the task reward,
   even when the other cases pass? Restore it afterward.
4. Change `TEST_TIMEOUT_S`, or run with `--test-timeout 1`. The timeout is per
   execution call, not a limit on model generation or sandbox startup.

Choose how many samples to evaluate with the same command:

```bash
./run-experiment 1 run --limit 3
./run-experiment 1 run --limit 5
./run-experiment 1 run --samples-file path/to/samples.jsonl --limit 2
./run-experiment 1 run --limit 2 --repeats 3
./run-experiment 1 run --help
```

`--limit` defaults to 1 and selects the first N samples in file order. The file
contains factorial, sum, vowel counting, sorted unique values, and palindrome
detection. `--samples-file` lets you use another file with the same structure.
`--repeats` defaults to 1; `--limit 2 --repeats 3` generates six scored responses,
with a fresh sandbox for each response. Invalid or empty sample files and limits
larger than the available sample count are rejected before Gym starts.

These are synthetic smoke tests, not a benchmark of model coding ability.
`--model` and `--output-dir` are also supported. Run one module at a time; the
local Gym head server uses port 11900.


## Saved artifacts

Each run creates `results/module_1_python/<UTC datetime>-model/`
(or `-golden` for a reference run). `--output-dir` overrides this location.
The terminal prints the path and saves its transcript in `console.log`.

Open the generated run `README.md` for the score, completion status, and links
to all evidence. Model runs keep Gym's scored rollouts, aggregate metrics,
failure records, prepared datasets, component logs, and native model-call
captures. Module-specific evidence remains alongside them. Failed runs keep
`run.json`, `error.log`, logs, and partial outputs without claiming a score.

See [the output layout](../../README.md#output-layout) for the file catalog.
