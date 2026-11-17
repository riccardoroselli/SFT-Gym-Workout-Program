"""Tests for src/request_parser.py: every data-v1 request reads back as its own meta; overlapping words; errors."""
import pytest

from evaluate import ROOT, read_jsonl
from make_dataset import NUMBER_WORDS
from request_parser import NUMBER_NAMES, RequestError, parse_request

KEYS = ["body_part", "equipment_group", "difficulty", "n_exercises"]


def test_every_data_v1_request_parses_to_its_meta():
    rows = [r for split in ("train", "val", "test") for r in read_jsonl(ROOT / f"data/processed/{split}.jsonl")]
    assert len(rows) == 2500
    wrong = [(r["messages"][1]["content"], r["meta"]) for r in rows
             if parse_request(r["messages"][1]["content"]) != {k: r["meta"][k] for k in KEYS}]
    assert wrong == []


def test_number_words_agree_with_make_dataset():
    assert all(NUMBER_NAMES[n - 1] == w for n, w in NUMBER_WORDS.items())


@pytest.mark.parametrize("text, part, value", [
    ("forearms workout, hard, free weights", "body_part", "forearms"),
    ("lower arms workout, hard, free weights", "body_part", "forearms"),
    ("upper arms workout, hard, free weights", "body_part", "arms"),
    ("biceps and triceps, hard, free weights", "body_part", "arms"),
    ("upper legs workout, hard, free weights", "body_part", "legs"),
    ("lower legs workout, hard, free weights", "body_part", "calves"),
    ("lats and upper back, hard, free weights", "body_part", "back"),
    ("core workout, hard, free weights", "body_part", "abs"),
    ("abs and core workout, hard, free weights", "body_part", "abs"),  # the same value twice is fine
    ("chest workout, hard, free weights", "equipment_group", "free weights"),
    ("chest workout, hard, weights", "equipment_group", "free weights"),
    ("chest workout, hard, gym machines", "equipment_group", "machines"),
    ("chest workout, hard, machines and cables", "equipment_group", "machines"),
    ("chest workout, hard, bodyweight exercises", "equipment_group", "bodyweight"),
    ("chest workout, hard, anything in the gym", "equipment_group", "any"),
    ("CHEST workout, Intense, FREE   WEIGHTS", "equipment_group", "free weights"),  # case and spacing
    ("light chest workout with bands", "difficulty", "easy"),
    ("chest workout, advanced level, bands", "difficulty", "hard"),
    ("chest workout, hard, bands", "n_exercises", 4),
    ("chest workout, hard, bands, 6 exercises", "n_exercises", 6),
    ("chest workout, hard, bands, three exercises", "n_exercises", 3),
])
def test_overlapping_words_and_counts(text, part, value):
    assert parse_request(text)[part] == value


@pytest.mark.parametrize("text, missing, conflicts, bad_count", [
    ("medium workout with free weights", ["body_part"], {}, None),
    ("chest please", ["equipment_group", "difficulty"], {}, None),
    ("", ["body_part", "equipment_group", "difficulty"], {}, None),
    ("chest and back workout, medium, free weights", [], {"body_part": ["chest", "back"]}, None),
    ("chest, easy or hard, bands", [], {"difficulty": ["easy", "hard"]}, None),
    ("chest, medium, bands, 3 or 4 exercises", [], {"n_exercises": [3, 4]}, None),
    ("chest, medium, bands, 7 exercises", [], {}, 7),
    ("chest, medium, bands, two exercises", [], {}, 2),
])
def test_errors(text, missing, conflicts, bad_count):
    with pytest.raises(RequestError) as e:
        parse_request(text)
    assert (e.value.missing, e.value.conflicts, e.value.bad_count) == (missing, conflicts, bad_count)
    detail = e.value.detail()
    assert detail["message"] == str(e.value) and detail["message"]
    assert "pecs" in detail["allowed"]["body_part"]["chest"] and "calisthenics" in detail["allowed"]["equipment_group"]["bodyweight"]
    assert "3-6" in detail["allowed"]["n_exercises"]
