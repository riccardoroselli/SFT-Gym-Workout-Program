"""Tests for src/compare_runs.py on fake runs (no model, no files)."""
import json

import pytest

from compare_runs import compare, is_refusal

CHEST = {"body_part": "chest", "equipment_group": "free weights", "difficulty": "medium", "n_exercises": 4, "feasible": True}
IMPOSSIBLE = {"body_part": "forearms", "equipment_group": "bands", "difficulty": "easy", "n_exercises": 3, "feasible": False}
NAMES = ["barbell bench press", "dumbbell bench press", "barbell incline bench press", "barbell decline bench press"]
GOOD = json.dumps({"workout": [{"name": n, "sets": 3, "reps": "8-12"} for n in NAMES]})
WRONG_SETS = json.dumps({"workout": [{"name": n, "sets": 4, "reps": "8-12"} for n in NAMES]})  # reward 0.91, not all_correct
ERROR = json.dumps({"error": "not enough exercises"})


def run(*rows):
    return [{"request": f"request {i}", "meta": meta, "output": out} for i, (meta, out) in enumerate(rows)]


def test_paired_counts():
    a = run((CHEST, GOOD), (CHEST, GOOD), (CHEST, WRONG_SETS), (IMPOSSIBLE, GOOD), (CHEST, GOOD))
    b = run((CHEST, GOOD), (CHEST, WRONG_SETS), (CHEST, GOOD), (IMPOSSIBLE, ERROR), (CHEST, ERROR))
    s = compare(a, b, resamples=500)
    assert s["identical"] == 1
    assert (s["higher"], s["lower"], s["equal"]) == (2, 2, 0)  # row 1 lower, 2 higher, 3 higher (refused), 4 lower
    assert (s["gained"], s["lost"]) == (2, 2)
    assert s["false_refusals"] == (0, 1) and s["feasible"] == 4
    assert s["all_correct"][0] == 0.0
    mean, low, high = s["reward"]
    assert low <= mean <= high


def test_identical_runs_have_zero_interval():
    a = run((CHEST, GOOD), (CHEST, WRONG_SETS))
    s = compare(a, a, resamples=100)
    assert s["identical"] == 2 and s["reward"] == (0.0, 0.0, 0.0)


def test_seed_makes_it_reproducible():
    a = run((CHEST, GOOD), (CHEST, WRONG_SETS), (CHEST, GOOD))
    b = run((CHEST, WRONG_SETS), (CHEST, GOOD), (CHEST, ERROR))
    assert compare(a, b, resamples=200) == compare(a, b, resamples=200)


def test_different_requests_rejected():
    a = run((CHEST, GOOD))
    b = [{**a[0], "request": "something else"}]
    with pytest.raises(ValueError, match="same requests"):
        compare(a, b)


def test_is_refusal():
    assert is_refusal(ERROR) and not is_refusal(GOOD) and not is_refusal("not json")
