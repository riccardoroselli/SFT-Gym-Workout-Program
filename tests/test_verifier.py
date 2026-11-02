"""Tests for src/verifier.py. Run from the project root: pytest"""
import json
import re
from pathlib import Path

from verifier import is_feasible, verify

SPEC = (Path(__file__).resolve().parent.parent / "spec.md").read_text()

# A valid medium chest free-weights workout (the spec.md example names).
CHEST_FW_MEDIUM = {"body_part": "chest", "equipment_group": "free weights", "difficulty": "medium", "n_exercises": 4}
CHEST_NAMES = ["barbell bench press", "dumbbell bench press", "barbell incline bench press", "barbell decline bench press"]
IMPOSSIBLE = {"body_part": "forearms", "equipment_group": "bands", "difficulty": "easy", "n_exercises": 3}


def answer(names, sets=3, reps="8-12"):
    return json.dumps({"workout": [{"name": n, "sets": sets, "reps": reps} for n in names]})


def test_perfect_answer_passes():
    r = verify(CHEST_FW_MEDIUM, answer(CHEST_NAMES))
    assert r["all_correct"] and r["reward"] == 1.0


def test_made_up_exercise_name():
    r = verify(CHEST_FW_MEDIUM, answer(CHEST_NAMES[:3] + ["super chest blaster 3000"]))
    assert r["names_exist"] == 0.75
    assert r["body_part_ok"] == 0.75  # an unknown name can't have the right body part
    assert not r["all_correct"] and r["reward"] < 1.0


def test_wrong_body_part():
    r = verify(CHEST_FW_MEDIUM, answer(CHEST_NAMES[:3] + ["barbell bent over row"]))  # back
    assert r["names_exist"] == 1.0 and r["body_part_ok"] == 0.75
    assert not r["all_correct"]


def test_wrong_equipment():
    r = verify(CHEST_FW_MEDIUM, answer(CHEST_NAMES[:3] + ["lever chest press"]))  # machine
    assert r["body_part_ok"] == 1.0 and r["equipment_ok"] == 0.75
    assert not r["all_correct"]


def test_duplicate_exercise():
    r = verify(CHEST_FW_MEDIUM, answer(CHEST_NAMES[:3] + [CHEST_NAMES[0]]))
    assert r["no_duplicates"] is False
    assert not r["all_correct"]


def test_wrong_count():
    r = verify(CHEST_FW_MEDIUM, answer(CHEST_NAMES[:3]))
    assert r["count_ok"] is False
    assert not r["all_correct"]


def test_wrong_sets_reps():
    r = verify(CHEST_FW_MEDIUM, answer(CHEST_NAMES, sets=4, reps="5-8"))  # hard scheme on a medium request
    assert r["sets_reps_ok"] is False
    assert not r["all_correct"]


def test_advanced_exercise_in_easy_workout():
    request = {"body_part": "chest", "equipment_group": "bodyweight", "difficulty": "easy", "n_exercises": 3}
    r = verify(request, answer(["push-up", "incline push-up", "clap push up"], sets=2, reps="12-15"))
    assert r["advanced_ok"] is False
    assert r["names_exist"] == 1.0 and r["sets_reps_ok"] is True
    assert not r["all_correct"]


def test_invalid_json():
    r = verify(CHEST_FW_MEDIUM, '{"workout": [{"name": "barbell bench press", "sets": 3,')
    assert r["parse_ok"] is False and r["reward"] == 0.0


def test_json_in_fences_parses_but_not_strict():
    r = verify(CHEST_FW_MEDIUM, "Here is your workout:\n```json\n" + answer(CHEST_NAMES) + "\n```\nEnjoy!")
    assert r["parse_ok"] is True and r["strict_format"] is False
    assert r["names_exist"] == 1.0 and not r["all_correct"]
    assert r["reward"] == 0.9  # full content, no format bonus


def test_error_for_impossible_request_is_correct():
    assert not is_feasible(IMPOSSIBLE)
    r = verify(IMPOSSIBLE, '{"error": "Not enough forearm band exercises."}')
    assert r["error_correct"] is True and r["all_correct"] and r["reward"] == 1.0


def test_error_for_possible_request_is_wrong():
    assert is_feasible(CHEST_FW_MEDIUM)
    r = verify(CHEST_FW_MEDIUM, '{"error": "Not enough exercises."}')
    assert r["error_correct"] is None  # not an impossible request
    assert not r["all_correct"] and r["reward"] == 0.1  # only the format bonus


# The two examples written in spec.md, read from the file so the tests follow the spec.
def spec_json_block(containing):
    blocks = re.findall(r"```json\n(.*?)```", SPEC, re.DOTALL)
    return next(b for b in blocks if containing in b)


def test_spec_valid_example():
    request = {"body_part": "chest", "equipment_group": "free weights", "difficulty": "medium"}  # no count -> 4
    assert verify(request, spec_json_block("barbell bench press"))["all_correct"]


def test_spec_error_example():
    assert verify(IMPOSSIBLE, spec_json_block('"error"'))["all_correct"]
