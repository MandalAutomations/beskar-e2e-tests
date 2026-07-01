# beskar-e2e-tests

A small, self-contained Python project used as an **example workflow job** for the
[Beskar AI](https://github.com/MandalAutomations/Beskar) control plane. It
validates the repo-committed [workflow-jobs](https://github.com/MandalAutomations/Beskar/blob/main/docs/jobs.md#workflow-jobs-65)
feature (#65) end-to-end: Beskar clones this repo and runs the steps in
[`orchestra.yml`](./orchestra.yml) on an approved runner image inside the gVisor
sandbox.

It runs a **real GPU training loop** (PyTorch, full-batch gradient descent on a
convex quadratic) to exercise Beskar's GPU passthrough end to end. The problem is
convex with a sub-critical step size, so the loss is strictly decreasing and the
run stays fast and deterministic while genuinely using CUDA.

By default a GPU is **required**: `trainer.py` exits non-zero if CUDA is absent,
so a passing Beskar run really did touch the GPU. Set `REQUIRE_GPU=0` to allow a
CPU fallback for local development.

## What it does

| File | Role |
|------|------|
| `orchestra.yml` | The workflow: show env (`nvidia-smi`) → install deps → run unit tests → train → evaluate → verify artifacts. |
| `trainer.py` | A deterministic convex GD loop on the GPU; writes `model.json` (incl. the CUDA device) to `OUTPUT_DIR`. |
| `evaluate.py` | Reads the model, writes `eval.json`, exits non-zero if accuracy is below threshold (exercises fail-fast steps). |
| `tests/test_trainer.py` | Unit tests for the trainer logic, run as a workflow step. |

Outputs are written under `OUTPUT_DIR` (default `/scratch/workspace/artifacts`),
the job's writable, encrypted, ephemeral scratch.

## Run it locally

```bash
pip install -r requirements.txt          # installs torch (CPU wheel is fine locally)
pytest -q
# No GPU on your laptop? REQUIRE_GPU=0 lets it fall back to CPU:
REQUIRE_GPU=0 OUTPUT_DIR=./artifacts python trainer.py
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
  "repo_ref": "main",
  "gpu_min_vram_gb": 1
}
```

`gpu_min_vram_gb > 0` tells the scheduler to lease a GPU (`NVIDIA_VISIBLE_DEVICES`),
so `nvidia-smi` and `torch.cuda` see the card. Or use the portal's
**New job → Workflow (repo)** form with the same repo URL.
The job's logs show each step (parsed from `::orchestra:step::` markers) and the
Steps panel reflects progress.

## Tuning the run

Set these in `orchestra.yml`'s `env`:

- `EPOCHS` (default `3`) — more epochs → lower final loss.
- `SEED` (default `42`) — changes the init/target/jitter; runs stay deterministic.
- `REQUIRE_GPU` (default `1`) — `1` fails the job if no CUDA device; `0` allows CPU.
- `OUTPUT_DIR` (default `/scratch/workspace/artifacts`) — where artifacts are written.
