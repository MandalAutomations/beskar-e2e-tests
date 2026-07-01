import math

from evaluate import ACCURACY_THRESHOLD
from trainer import train


def test_train_is_deterministic():
    # Same seed on the same device is bitwise-reproducible.
    assert train(3, seed=42) == train(3, seed=42)


def test_seed_changes_accuracy_jitter():
    assert train(3, seed=1) != train(3, seed=2)


def test_loss_strictly_decreases():
    # Convex quadratic + sub-critical step size ⇒ loss falls every epoch.
    losses = [row["loss"] for row in train(8, seed=0)["history"]]
    assert all(b < a for a, b in zip(losses, losses[1:]))


def test_history_length_matches_epochs():
    result = train(7)
    assert result["epochs"] == 7
    assert len(result["history"]) == 7


def test_default_epochs_clear_eval_threshold():
    # The orchestra.yml default (EPOCHS=3) should produce a passing run.
    assert train(3, seed=42)["final_accuracy"] >= ACCURACY_THRESHOLD


def test_result_reports_device():
    # The artifact records where it trained so the job logs prove GPU usage.
    result = train(3, seed=42)
    assert result["device"] in {"cuda", "cpu"} or result["device"].startswith("cuda")
    assert "cuda_available" in result["cuda"]


def test_accuracy_in_unit_range():
    for row in train(5, seed=3)["history"]:
        assert 0.0 <= row["accuracy"] <= 0.99
        assert math.isfinite(row["loss"])
