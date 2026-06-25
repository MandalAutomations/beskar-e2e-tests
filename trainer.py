"""A tiny, dependency-free "training" job for exercising Beskar workflow jobs.

No GPU and no ML libraries, so the workflow runs anywhere the platform runner
image does. It reads ``EPOCHS``/``SEED`` from the environment, runs a
deterministic mock training loop with a strictly decreasing loss, and writes a
model artifact under ``OUTPUT_DIR`` (the job's ``/scratch/workspace``).
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

DEFAULT_OUTPUT_DIR = "/scratch/workspace/artifacts"


def train(epochs: int, seed: int = 0) -> dict:
    """Run a deterministic mock training loop and return the result.

    Loss decays geometrically (always strictly decreasing); accuracy tracks
    ``1 - loss`` with a small seeded jitter so the seed is meaningful while the
    loss curve stays monotonic and testable.
    """
    rng = random.Random(seed)
    loss = 1.0
    history = []
    for epoch in range(1, epochs + 1):
        loss = round(loss * 0.6, 4)
        accuracy = round(max(0.0, min(0.99, (1 - loss) + rng.uniform(-0.01, 0.01))), 4)
        history.append({"epoch": epoch, "loss": loss, "accuracy": accuracy})
    return {
        "epochs": epochs,
        "seed": seed,
        "final_loss": loss,
        "final_accuracy": history[-1]["accuracy"] if history else 0.0,
        "history": history,
    }


def main() -> None:
    epochs = int(os.environ.get("EPOCHS", "3"))
    seed = int(os.environ.get("SEED", "0"))
    out_dir = Path(os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[trainer] training for {epochs} epoch(s), seed={seed}")
    result = train(epochs, seed)
    for row in result["history"]:
        print(f"[trainer] epoch {row['epoch']}/{epochs} "
              f"loss={row['loss']} acc={row['accuracy']}")

    model_path = out_dir / "model.json"
    model_path.write_text(json.dumps(result, indent=2))
    print(f"[trainer] wrote {model_path}")


if __name__ == "__main__":
    main()
