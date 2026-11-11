"""Tests for src/probe_groups.py: request picking and the summary maths (no model needed)."""
import json

from evaluate import TRAIN, read_jsonl
from probe_groups import combo, pick_requests, summarize
from verifier import verify

CHEST = {"body_part": "chest", "equipment_group": "free weights", "difficulty": "medium", "n_exercises": 4, "feasible": True}
IMPOSSIBLE = {"body_part": "forearms", "equipment_group": "bands", "difficulty": "easy", "n_exercises": 3, "feasible": False}
NAMES = ["barbell bench press", "dumbbell bench press", "barbell incline bench press", "barbell decline bench press"]
GOOD = json.dumps({"workout": [{"name": n, "sets": 3, "reps": "8-12"} for n in NAMES]})
DUP = json.dumps({"workout": [{"name": n, "sets": 3, "reps": "8-12"} for n in NAMES[:3] + NAMES[:1]]})
ERROR = json.dumps({"error": "Not enough exercises."})


def rows(idx, meta, outputs, tokens=100):
    return [{"idx": idx, "meta": meta, "tokens": tokens, "truncated": tokens >= 256, "result": verify(meta, o)}
            for o in outputs]


def test_summary_maths():
    data = (rows(0, CHEST, [GOOD, GOOD])            # same reward twice: zero std, passes
            + rows(1, CHEST, [GOOD, DUP])           # rewards differ: std > 0, still passes (one correct)
            + rows(2, CHEST, [DUP, ERROR], tokens=256)  # no correct answer; one false refusal; both hit 256
            + rows(3, IMPOSSIBLE, [ERROR, GOOD]))   # one refusal out of two
    s = summarize(data)
    f = s["feasible n=4"]
    assert f["groups"] == 3 and f["answers"] == 6
    assert f["zero_std"] == round(1 / 3, 4)
    assert f["pass_at_g"] == round(2 / 3, 4)
    assert f["all_correct"] == 0.5
    assert f["false_refusal"] == round(1 / 6, 4) and f["refusal"] is None
    assert f["duplicates"] == round(2 / 5, 4)       # 2 of the 5 workout answers repeat a name
    assert f["hit_limit"] == 2 and f["tokens_max"] == 256
    r_good, r_dup, r_err = (verify(CHEST, o)["reward"] for o in (GOOD, DUP, ERROR))
    # pstdev of 2 values = half their gap
    assert abs(f["reward_std"] - (0 + abs(r_good - r_dup) / 2 + abs(r_dup - r_err) / 2) / 3) < 1e-4

    imp = s["impossible"]
    assert imp["refusal"] == 0.5 and imp["false_refusal"] is None
    assert imp["zero_std"] == 0.0 and imp["pass_at_g"] == 1.0
    assert s["all"]["groups"] == 4


def test_pick_requests():
    picked = pick_requests(read_jsonl(TRAIN))
    assert len(picked) == 40 and len({combo(r["meta"]) for r in picked}) == 40
    impossible = [r["meta"] for r in picked if not r["meta"]["feasible"]]
    assert len(impossible) == 8 and len({(m["body_part"], m["equipment_group"]) for m in impossible}) == 5
    for n in (3, 4, 5, 6):
        group = [r["meta"] for r in picked if r["meta"]["feasible"] and r["meta"]["n_exercises"] == n]
        assert len(group) == 8
        assert len({m["equipment_group"] for m in group}) == 5 and len({m["difficulty"] for m in group}) == 3
    assert pick_requests(read_jsonl(TRAIN)) == picked  # fixed seed: same 40 every time
