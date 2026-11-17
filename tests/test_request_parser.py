"""Tests for src/request_parser.py: every data-v1 request reads back as its own meta; overlapping words; aliases;
the model input rule; errors."""
import pytest

from evaluate import ROOT, read_jsonl
from make_dataset import NUMBER_WORDS, TEST_TEMPLATES, TRAIN_TEMPLATES
from request_parser import (CANONICAL_TEMPLATE, EXAMPLE_REQUEST, NUMBER_NAMES, SERVING_ALIASES, RequestError,
                            canonical_text, model_input, parse_request)

KEYS = ["body_part", "equipment_group", "difficulty", "n_exercises"]


def test_every_data_v1_request_parses_to_its_meta_without_aliases():
    rows = [r for split in ("train", "val", "test") for r in read_jsonl(ROOT / f"data/processed/{split}.jsonl")]
    assert len(rows) == 2500
    wrong = []
    for r in rows:
        text = r["messages"][1]["content"]
        req, notes = parse_request(text)
        if req != {k: r["meta"][k] for k in KEYS} or notes or model_input(text, req, notes) != text:
            wrong.append((text, r["meta"], notes))
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
    assert parse_request(text)[0][part] == value


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
    assert str(e.value) and detail["message"].startswith(str(e.value)) and EXAMPLE_REQUEST in detail["message"]
    assert detail["example"] == EXAMPLE_REQUEST and parse_request(EXAMPLE_REQUEST)[1] == []  # the example is valid
    assert "pecs" in detail["allowed"]["body_part"]["chest"] and "calisthenics" in detail["allowed"]["equipment_group"]["bodyweight"]
    assert "dumbbells" in detail["allowed"]["equipment_group"]["free weights"]  # aliases are listed too
    assert "3-6" in detail["allowed"]["n_exercises"]


# --- serving aliases ---
SPEC_WORDS = {"body_part": "chest", "equipment_group": "free weights", "difficulty": "easy"}


@pytest.mark.parametrize("alias", list(SERVING_ALIASES))
def test_every_alias(alias):
    part, value = SERVING_ALIASES[alias]
    others = ", ".join(w for p, w in SPEC_WORDS.items() if p != part)  # the other parts in spec words
    req, notes = parse_request(f"{alias} workout, {others}, 5 exercises")
    assert req[part] == value and len(notes) == 1 and notes[0].startswith(f"{alias} → {value} (")


@pytest.mark.parametrize("text, part, value, aliases_used", [
    ("forearm workout, easy, bands", "body_part", "forearms", ["forearm"]),  # not "arm"
    ("forearms workout, easy, bands", "body_part", "forearms", []),
    ("arm workout, easy, bands", "body_part", "arms", ["arm"]),
    ("calf workout, easy, bands", "body_part", "calves", ["calf"]),
    ("calves workout, easy, bands", "body_part", "calves", []),
    ("leg workout, easy, bands", "body_part", "legs", ["leg"]),
    ("lower legs workout, easy, bands", "body_part", "calves", []),  # not "leg(s)"
    ("lats and upper back, easy, bands", "body_part", "back", []),  # spec phrase, not the "lats" alias
    ("lats workout, easy, bands", "body_part", "back", ["lats"]),
    ("biceps and triceps, easy, bands", "body_part", "arms", []),  # spec phrase, not two aliases
    ("biceps, easy, bands", "body_part", "arms", ["biceps"]),
    ("ab workout, easy, bands", "body_part", "abs", ["ab"]),
    ("chest, easy, band", "equipment_group", "bands", ["band"]),
    ("chest, easy, bands", "equipment_group", "bands", []),
    ("chest, easy, resistance band", "equipment_group", "bands", ["resistance band"]),
    ("chest, easy, resistance bands", "equipment_group", "bands", []),
    ("chest, easy, machine", "equipment_group", "machines", ["machine"]),
    ("chest, easy, machines", "equipment_group", "machines", []),
    ("chest, easy, machines and cables", "equipment_group", "machines", []),  # not "machines" + "cables"
    ("chest, easy, cables", "equipment_group", "machines", ["cables"]),
    ("chest, easy, body weight", "equipment_group", "bodyweight", ["body weight"]),
    ("chest, easy, bodyweight exercises", "equipment_group", "bodyweight", []),
    ("chest, easy, dumbbells and barbells", "equipment_group", "free weights", ["dumbbells", "barbells"]),
    ("biceps and tricep day, easy, Dumbbells", "body_part", "arms", ["biceps", "tricep", "dumbbells"]),
])
def test_alias_overlaps_and_notes(text, part, value, aliases_used):
    req, notes = parse_request(text)
    assert req[part] == value
    assert [n.split(" → ")[0] for n in notes] == aliases_used


def test_note_lists_the_whole_equipment_group():
    _, notes = parse_request("biceps workout, easy, dumbbells, 5 exercises")
    assert notes == ["biceps → arms (catalog body part: upper arms)",
                     "dumbbells → free weights (dumbbell, barbell, ez barbell, olympic barbell, kettlebell, trap bar, "
                     "weighted, medicine ball)"]


def test_alias_conflict_is_still_a_conflict():
    with pytest.raises(RequestError) as e:
        parse_request("biceps, easy, dumbbells and bands")
    assert e.value.conflicts == {"equipment_group": ["free weights", "bands"]}


def test_no_equipment_is_not_an_alias():
    with pytest.raises(RequestError) as e:
        parse_request("chest workout, easy, no equipment")
    assert e.value.missing == ["equipment_group"]


# --- what the model receives ---
def test_model_input_rule():
    text = "biceps workout, easy, dumbbells, 5 exercises"
    req, notes = parse_request(text)
    assert model_input(text, req, notes) == "Arms workout, easy difficulty, free weights, 5 exercises"
    spec_text = "Arms workout, easy difficulty, free weights, 5 exercises"
    assert model_input(spec_text, *parse_request(spec_text)) == spec_text  # spec words only: unchanged


def test_canonical_text_is_a_train_template_and_reads_back_for_every_combination():
    from make_dataset import all_combos
    assert CANONICAL_TEMPLATE in TRAIN_TEMPLATES and CANONICAL_TEMPLATE not in TEST_TEMPLATES
    for req in all_combos():  # 480 combinations; "abs" stands for abs/core
        assert parse_request(canonical_text(req)) == (req, []), req
