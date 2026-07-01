"""A tiny GPU "training" job for exercising Beskar GPU workflow jobs.

This runs a *real* PyTorch optimisation on the GPU: full-batch gradient descent
on a strictly convex quadratic. Because the problem is convex and the step size
is below ``1 / max_curvature``, the loss is guaranteed to decrease every epoch,
so the run stays deterministic and testable while genuinely exercising CUDA.

It reads ``EPOCHS``/``SEED`` from the environment, trains on the GPU, and writes
a model artifact under ``OUTPUT_DIR`` (the job's ``/scratch/workspace``). The
artifact records the CUDA device it ran on, so the job logs prove the GPU was
used end to end.

By default a GPU is *required*: if CUDA is unavailable the job exits non-zero, so
a passing Beskar run really did touch the GPU. Set ``REQUIRE_GPU=0`` to allow a
CPU fallback for local development.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

DEFAULT_OUTPUT_DIR = "/scratch/workspace/artifacts"

# Problem size and step size. LR is comfortably below 1 / max_curvature (max
# curvature is 1.2 below), which makes full-batch GD on this convex quadratic
# strictly decrease the loss every epoch — see module docstring.
DIM = 16
LR = 0.15


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_device(require_gpu: bool) -> torch.device:
    """Pick the training device, honouring the GPU requirement.

    Returns CUDA when available. If CUDA is missing and ``require_gpu`` is set,
    exits the process non-zero (fails the Beskar job) rather than silently
    training on the CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_gpu:
        print(
            "[trainer] no CUDA device available and REQUIRE_GPU is set; "
            "refusing to fall back to CPU. Submit the job with gpu_min_vram_gb "
            "> 0, or set REQUIRE_GPU=0 for local CPU dev.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("[trainer] CUDA unavailable; falling back to CPU (REQUIRE_GPU=0)")
    return torch.device("cpu")


def _cuda_info() -> dict:
    info = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info


def train(epochs: int, seed: int = 0, device: torch.device | None = None) -> dict:
    """Run a deterministic convex GD loop on ``device`` and return the result.

    Minimises ``loss(w) = sum a_i * (w_i - t_i)^2`` (a strictly convex, separable
    quadratic) with full-batch gradient descent. The loss is strictly decreasing;
    accuracy tracks the fraction of the initial loss removed so far, with a small
    seeded jitter so the seed is meaningful while the curve stays monotonic.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # A per-device generator gives run-to-run bitwise-identical results for the
    # same seed (all ops below are elementwise, hence deterministic on CUDA).
    gen = torch.Generator(device=device).manual_seed(seed)

    # Curvature in [0.8, 1.2], a random optimum t, and a random start w.
    a = 0.8 + 0.4 * torch.rand(DIM, generator=gen, device=device)
    t = torch.randn(DIM, generator=gen, device=device)
    w = torch.randn(DIM, generator=gen, device=device)

    def loss_of(weights: torch.Tensor) -> torch.Tensor:
        return (a * (weights - t) ** 2).sum()

    initial_loss = loss_of(w)

    history = []
    for epoch in range(1, epochs + 1):
        grad = 2.0 * a * (w - t)
        w = w - LR * grad
        loss = loss_of(w)

        # Fraction of the initial loss removed so far, in (0, 1).
        fraction = (1.0 - loss / initial_loss).item()
        jitter = torch.rand((), generator=gen, device=device).item() * 0.02 - 0.01
        accuracy = round(max(0.0, min(0.99, fraction + jitter)), 4)
        history.append({"epoch": epoch, "loss": round(loss.item(), 6), "accuracy": accuracy})

    return {
        "epochs": epochs,
        "seed": seed,
        "device": str(device),
        "cuda": _cuda_info(),
        "final_loss": history[-1]["loss"] if history else round(initial_loss.item(), 6),
        "final_accuracy": history[-1]["accuracy"] if history else 0.0,
        "history": history,
    }


def main() -> None:
    epochs = int(os.environ.get("EPOCHS", "3"))
    seed = int(os.environ.get("SEED", "0"))
    require_gpu = _truthy(os.environ.get("REQUIRE_GPU", "1"))
    out_dir = Path(os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(require_gpu)
    print(f"[trainer] training for {epochs} epoch(s), seed={seed} on {device}")
    if device.type == "cuda":
        print(f"[trainer] GPU: {torch.cuda.get_device_name(0)} "
              f"(torch {torch.__version__}, CUDA {torch.version.cuda})")

    result = train(epochs, seed, device=device)
    for row in result["history"]:
        print(f"[trainer] epoch {row['epoch']}/{epochs} "
              f"loss={row['loss']} acc={row['accuracy']}")

    model_path = out_dir / "model.json"
    model_path.write_text(json.dumps(result, indent=2))
    print(f"[trainer] wrote {model_path}")


if __name__ == "__main__":
    main()
