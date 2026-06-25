# beskar-e2e-tests

A small, self-contained Python project used as an **example workflow job** for the
[Beskar AI](https://github.com/MandalAutomations/Beskar) control plane. It
validates the repo-committed [workflow-jobs](https://github.com/MandalAutomations/Beskar/blob/main/docs/jobs.md#workflow-jobs-65)
feature (#65) end-to-end: Beskar clones this repo and runs the steps in
[`orchestra.yml`](./orchestra.yml) on an approved runner image inside the gVisor
sandbox.

It deliberately uses **no GPU and no ML dependencies** (only `pytest`), so the
workflow runs anywhere the runner image does and stays fast and deterministic.

## What it does

| File | Role |
|------|------|
| `orchestra.yml` | The workflow: show env → install deps → run unit tests → train → evaluate → verify artifacts. |
| `trainer.py` | A deterministic mock "training" loop; writes `model.json` to `OUTPUT_DIR`. |
| `evaluate.py` | Reads the model, writes `eval.json`, exits non-zero if accuracy is below threshold (exercises fail-fast steps). |
| `tests/test_trainer.py` | Unit tests for the trainer logic, run as a workflow step. |

Outputs are written under `OUTPUT_DIR` (default `/scratch/workspace/artifacts`),
the job's writable, encrypted, ephemeral scratch.

## Run it locally

```bash
pip install -r requirements.txt
pytest -q
OUTPUT_DIR=./artifacts python trainer.py
OUTPUT_DIR=./artifacts python evaluate.py
```

## Run it on Beskar

Submit a workflow job pointing at this repo (see
[jobs.md](https://github.com/MandalAutomations/Beskar/blob/main/docs/jobs.md#workflow-jobs-65)):

```http
POST /api/v1/jobs
Authorization: Bearer <customer_access_token>

{
  "repo_url": "https://github.com/MandalAutomations/beskar-e2e-tests",
  "repo_ref": "main"
}
```

Or use the portal's **New job → Workflow (repo)** form with the same repo URL.
The job's logs show each step (parsed from `::orchestra:step::` markers) and the
Steps panel reflects progress.

## Tuning the run

Set these in `orchestra.yml`'s `env`:

- `EPOCHS` (default `3`) — more epochs → lower final loss.
- `SEED` (default `42`) — changes the accuracy jitter; runs stay deterministic.
- `OUTPUT_DIR` (default `/scratch/workspace/artifacts`) — where artifacts are written.
