"""Tests for src/analyze_errors.py helpers."""
import json

from analyze_errors import closest, duplicates, first_failed, strip_variant
from verifier import verify

CHEST = {"body_part": "chest", "equipment_group": "free weights", "difficulty": "medium", "n_exercises": 4, "feasible": True}
IMPOSSIBLE = {"body_part": "forearms", "equipment_group": "bands", "difficulty": "easy", "n_exercises": 3, "feasible": False}
NAMES = ["barbell bench press", "dumbbell bench press", "barbell incline bench press", "barbell decline bench press"]


def answer(names, sets=3, reps="8-12"):
    return json.dumps({"workout": [{"name": n, "sets": sets, "reps": reps} for n in names]})


def test_first_failed_order():
    check = lambda meta, out: first_failed(meta, verify(meta, out))
    assert check(CHEST, answer(NAMES)) == "passed"
    assert check(CHEST, "not json") == "parse"
    assert check(CHEST, answer(NAMES[:3] + ["made up press"], sets=4)) == "names"  # names before sets/reps
    assert check(CHEST, answer(NAMES, sets=4)) == "sets/reps"
    assert check(IMPOSSIBLE, answer(["made up curl"] * 3)) == "error handling"  # not "names"


def test_closest_kinds():
    assert closest("Barbell  Bench Press")[:2] == ("case/spacing", "barbell bench press")
    assert closest("dumbbell flyes")[:2] == ("near-miss", "dumbbell fly")
    assert closest("super chest blaster 3000")[0] == "invented"


def test_duplicates_exact():
    assert duplicates(NAMES) == (0, False)
    assert duplicates(["a", "a", "a", "b"]) == (2, False)  # 2 extra copies, not 3


def test_duplicates_near():
    assert strip_variant("Dumbbell Rear Fly v. 2") == "dumbbell rear fly"
    assert duplicates(["dumbbell rear fly", "dumbbell rear fly v. 2"]) == (0, True)
    assert duplicates(["push-up (male)", "push-up (female)"]) == (0, True)
    assert duplicates(["dumbbell incline fly", "dumbbell incline fly on exercise ball",
                       "dumbbell incline fly on exercise ball"]) == (1, True)
    assert duplicates(["dumbbell fly", "dumbbell incline fly"]) == (0, False)  # different exercises
