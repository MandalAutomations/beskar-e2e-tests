"""Evaluate the trained artifact and write an eval report (a later workflow step).

Reads ``OUTPUT_DIR/model.json`` produced by ``trainer.py``, writes ``eval.json``
beside it, and exits non-zero if the run didn't clear the accuracy bar — which
fails the Beskar job, demonstrating fail-fast step semantics.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ACCURACY_THRESHOLD = 0.5


def main() -> int:
    out_dir = Path(os.environ.get("OUTPUT_DIR", "/scratch/workspace/artifacts"))
    model_path = out_dir / "model.json"
    if not model_path.exists():
        print(f"[evaluate] missing artifact: {model_path}", file=sys.stderr)
        return 1

    model = json.loads(model_path.read_text())
    passed = model["final_accuracy"] >= ACCURACY_THRESHOLD
    report = {
        "final_accuracy": model["final_accuracy"],
        "final_loss": model["final_loss"],
        "threshold": ACCURACY_THRESHOLD,
        "passed": passed,
    }
    (out_dir / "eval.json").write_text(json.dumps(report, indent=2))
    print(f"[evaluate] accuracy={report['final_accuracy']} "
          f"threshold={ACCURACY_THRESHOLD} passed={passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
