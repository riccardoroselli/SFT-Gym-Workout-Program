"""Tests for the score step of src/evaluate.py (no model needed)."""
import json

from evaluate import score

CHEST = {"body_part": "chest", "equipment_group": "free weights", "difficulty": "medium", "n_exercises": 4}
GOOD = json.dumps({"workout": [{"name": n, "sets": 3, "reps": "8-12"} for n in
                               ["barbell bench press", "dumbbell bench press",
                                "barbell incline bench press", "barbell decline bench press"]]})


def write_outputs(path, outputs, truncated):
    path.write_text("".join(json.dumps({"meta": CHEST, "output": o, "truncated": t}) + "\n"
                            for o, t in zip(outputs, truncated)))


def test_score_summary_and_table(tmp_path):
    write_outputs(tmp_path / "fake.jsonl", [GOOD, "<think>\n</think>\nnot json"], [False, True])
    s = score("fake", outputs_dir=tmp_path, results_dir=tmp_path)
    assert s["n"] == 2 and s["all_correct"] == 0.5 and s["reward"] == 0.5
    assert s["has_think"] == 0.5 and s["truncated"] == 0.5
    assert s["names_exist"] == 1.0      # the unparseable answer has None, so it's skipped
    assert s["error_correct"] is None   # no impossible requests here
    detail = json.loads((tmp_path / "fake.json").read_text())
    assert len(detail["examples"]) == 2

    score("fake", outputs_dir=tmp_path, results_dir=tmp_path)  # re-score: row replaced, not added
    table = (tmp_path / "results.md").read_text().splitlines()
    assert len(table) == 3 and table[2].startswith("| fake |")
