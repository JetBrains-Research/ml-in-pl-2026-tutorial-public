# Module 4: repair IDE inspections through integrated MCP

[All experiment modules](../../README.md) · [Run artifacts](../../README.md#output-layout)

## Start here

Open [workshop.py](workshop.py), then run from `tutorial-modules/module-2/`:

```bash
./run-experiment 4 run
```

Learn an **inspection -> edit -> inspection loop**, then distinguish agent
observations from an independent grade. The central script contains the actual
two-sandbox evaluation and runner. `.env` selects Gemma through LiteLLM; this
module uses the generic PyCharm MCP image in Google Artifact Registry, validated
through the workshop orchestrator.

## Follow the code

Read `evaluate()` for the episode flow, then `main()` for configuration and
reporting. The same four stages appear in Modules 1–4:

1. **`prepare_task()`** stages the public broken files and waits for PyCharm to
   detect problems in both. Agent and fresh grader sandboxes start concurrently;
   both baseline checks run concurrently before the model starts. The image
   supplies the IDE and MCP server, and contains no exercise files.
2. **`run_task()`** applies the reference repair or calls the bounded model loop
   with only inspect, read, replace and reformat tools. `ScopedInspectionConnection`
   restricts calls to the two task files. Observations are real PyCharm diagnostics
   delivered through MCP and IDEGym; the model has no access to the grader.
3. **`grade_task()`** receives the two submitted sources after the agent closes.
   `export_submission()` bounds and validates those files. The waiting fresh
   grader receives sources, checks final diagnostics and verifies behavior.
   `evaluate()` also checks the saved trajectory for successful inspections of
   both files before and after editing. All reward conditions must pass.
4. **`report_results()`** saves the summary and prints reward, individual checks
   and diagnostic counts. `sandboxes.json` records distinct sandbox IDs.

`evaluate()` visibly composes prepare -> run -> export -> close agent -> grade.
The shared `concurrent_sandboxes()` helper owns startup and cleanup. Only source
files cross the submission boundary; agent IDE settings, packages and processes
cannot change the grader's environment. Read `task.py` for the fixture and
baseline, and `save_verification()` for the detailed artifact report.

## Read the output and try a change

The terminal identifies the task/model/interface, shows agent setup and model
turns, announces the fresh grader, and presents individual checks and diagnostic
counts. Waiting statuses appear every 30 seconds. `--run-timeout` bounds the
complete evaluation (default 1,800 seconds); the two IDE startups overlap.
`--golden` validates the reference repair without model calls.

Compare `inspections-before.json`, `trajectory.jsonl` and `inspections-after.json`.
Find the model's before/after inspection calls and match them to the independent
grade. Check `sandboxes.json` to confirm two distinct sandboxes. Try
`--step-limit 4`: explain whether the model completed edits but failed to inspect
both files afterwards, and which reward condition captures that difference.

A model diagnoses and repairs a two-file Python pricing project using PyCharm's
bundled MCP server. The harness calls `pycharm_get_file_problems` independently
before the model runs in its sandbox, then grades the submitted files in a fresh disposable sandbox.

## Task and score

The harness installs the fixture from `task.py` before the episode. `pricing.py`
contains a misspelled variable that leaves its discount parameter unused;
`receipt.py` has a missing colon. The contract is to calculate discounted totals
and format receipts without changing public signatures.

Reward is 1 only when all these conditions hold:

- The independent baseline query finds at least one diagnostic in each file.
- The fresh grader detects the broken fixture before transfer, then its independent post-transfer queries return no errors or warnings in either file.
- External behavior checks pass, both files remain present, and no inspection
  suppression markers were added to them.
- The model successfully calls the inspection tool for both files before its
  first edit and again after its last edit. The reference run skips this condition.

A missing/invalid baseline aborts the evaluation rather than scoring an already
clean fixture. Tool failures and malformed inspection responses abort; they are
never interpreted as an empty diagnostic list. This grades these two exercise
files, not every possible project or IDE diagnostic.

The model receives four discovered integrated MCP tools: `get_file_problems`,
`get_file_text_by_path`, `replace_text_in_file`, and `reformat_file`. Calls are
restricted to the two task files. It cannot invoke terminal commands or change
inspection profiles through these tools. Setup, behavioral verification, and
artifact collection use the sandbox API privately; they are not model tools.
The model is not given the reference repair.

## Submission and fresh grading

The external harness overlaps startup, then runs the model and grading in order:

1. Start independent agent and grader sandboxes concurrently from the same pinned image.
2. Seed the broken fixture and validate baseline diagnostics in both sandboxes concurrently.
3. Run the model with the four scoped MCP tools (or apply the reference repair).
4. Export only `pricing.py` and `receipt.py`, save `final-files.json`, and close the agent sandbox.
5. Transfer only the submitted source files through a harness-owned execution command. IDE settings, packages, caches, and processes are not transferred.
6. Query final diagnostics and execute the behavioral verifier only in the grader. Inspect the model trace outside both sandboxes, record the reward, and close the grader.

The connection helper registers a separate IDEGym client for each sandbox; `sandboxes.json` records both client/server identities. Both servers use the same image and disable reuse. Model tools are bound only to the agent server. The grader starts alongside the agent and waits for submitted sources. The verifier source is never executed or installed in the agent sandbox.

Submission export rejects symlinks, directories, and files larger than 64 KiB each. Missing task files produce a failed grade. Grading executes submitted Python in the grader; this separates grading from the agent runtime, but does not isolate the verifier from deliberately hostile submitted code.

## Prepare or check the image

Participants use the existing validated image. Maintainers can rebuild it
through the workshop orchestrator’s Cloud Build backend:

```bash
./run-experiment 4 build-image
# Resume polling a submitted job:
./run-experiment 4 build-image --job JOB_NAME
./run-experiment 4 check
```

The image contains PyCharm 2026.1.1, IDEGym 0.12.0, and an empty project workspace.
The inspection exercise writes its task files at runtime; changing
this exercise does not require rebuilding the image. `check` seeds the inspection
fixture and confirms PyCharm reports diagnostics. `run --golden` also validates
the reference repair and fresh-sandbox grader.

## Run

From the repository root:

```bash
./run-experiment 4 run --golden
./run-experiment 4 run
# Gemma requires the reasoning-effort parameter to be omitted:
./run-experiment 4 run \
  --model vertex_ai/google/gemma-4-26b-a4b-it-maas --reasoning-effort omit
```

The default is `vertex_ai/google/gemma-4-26b-a4b-it-maas`, with reasoning effort
omitted and 20 turns.
The default image is
`europe-west4-docker.pkg.dev/gke-dev-dws-jbr/idegym/ml-in-pl-workshop-pycharm-mcp:5fa379b2f8d4e74aa9fba89eac673fd5`.
The image is fixed in `experiments/images.py` and this module's `config.yaml`.
No image build is needed for participants. Maintainer builds submit `image.yaml` through IDEGym. Its
pinned Dockerfile uses staged plugin assets and an empty workspace in GCS; no
local build, Docker installation, or source checkout is needed. The command
automatically pushes to the orchestrator’s configured Google registry and saves
the successful tag in `.runtime/module_4_jetbrains_mcp/image.json` as
a build record for maintainers. To use the new tag, update the image constant
and this module's config, then rebuild the runner.
Each run starts two fresh sandboxes concurrently and deletes each through its context-manager cleanup, including on exceptions. Both PyCharm startups and baseline checks overlap.
With local Python, run evaluations one at a time because Gym uses port 11900.
Separate Docker runner containers isolate these ports and can run concurrently.


## Saved artifacts

Each run creates `results/module_4_jetbrains_mcp/<UTC datetime>-model/`
(or `-golden` for a reference run). `--output-dir` overrides this location.
The terminal prints the path and saves its transcript in `console.log`.

Open the generated run `README.md` for the score, completion status, and links
to all evidence. Model runs keep Gym's scored rollouts, aggregate metrics,
failure records, prepared datasets, component logs, and native model-call
captures. Module-specific evidence remains alongside them. Failed runs keep
`run.json`, `error.log`, logs, and partial outputs without claiming a score.

See [the output layout](../../README.md#output-layout) for the file catalog.
