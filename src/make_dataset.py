"""Generate the SFT dataset: data/processed/{train,val,test}.jsonl.

Each line: {"messages": [system, user, assistant], "meta": {...}}.
Test uses only held-out request combinations (body_part, equipment_group, difficulty, n_exercises),
so it measures generalization to requests never seen in training.

    python src/make_dataset.py
"""
import json
import random
import re
from pathlib import Path

from spec import EQUIPMENT_GROUPS, SETS_REPS
from verifier import is_feasible, matching_exercises, verify

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/processed"
SEED = 42
SIZES = {"train": 2000, "val": 200, "test": 300}
TEST_COMBO_SHARE = 0.15
IMPOSSIBLE_SHARE = 0.05
OMIT_COUNT_SHARE = 0.5  # of n=4 requests, how many leave the count out
SYSTEM = "You create gym workouts from the user's request. Answer only with JSON."

SYNONYM_SHARE = 0.2  # how often a body part is said with a synonym instead of its friendly name
BODY_WORDS = {  # canonical name (spec.py BODY_PARTS key) -> (friendly names, occasional synonyms)
    "chest": (["chest"], ["pecs"]),
    "back": (["back"], ["lats and upper back"]),
    "shoulders": (["shoulders"], ["delts"]),
    "arms": (["arms"], ["upper arms", "biceps and triceps"]),
    "forearms": (["forearms"], ["lower arms"]),
    "legs": (["legs"], ["upper legs", "thighs"]),
    "calves": (["calves"], ["lower legs"]),
    "abs": (["abs", "core"], ["abdominals", "midsection", "waist"]),
}
EQUIP_WORDS = {
    "free weights": ["free weights", "weights"],
    "machines": ["machines", "gym machines", "machines and cables"],
    "bodyweight": ["bodyweight", "bodyweight exercises", "calisthenics"],
    "bands": ["bands", "resistance bands", "elastic bands"],
    "any": ["any equipment", "whatever equipment", "anything in the gym"],
}
DIFF_WORDS = {  # {diff}: adjective in front of "workout" ("a light chest workout")
    "easy": ["easy", "beginner", "light"],
    "medium": ["medium", "intermediate", "moderate"],
    "hard": ["hard", "advanced", "intense"],
}
LEVEL_WORDS = {  # {level}: always followed by "level" or "difficulty" ("at beginner level")
    "easy": ["easy", "beginner"],
    "medium": ["medium", "intermediate"],
    "hard": ["hard", "advanced"],
}
NUMBER_WORDS = {3: "three", 4: "four", 5: "five", 6: "six"}
# (template, count phrase); {c} becomes the count phrase, or "" when the count is left out.
TRAIN_TEMPLATES = [
    ("{body} workout, {level} difficulty, {equip}{c}", ", {n} exercises"),
    ("Give me a {diff} {body} workout using {equip}{c}.", " with {n} exercises"),
    ("I want to train my {body} today with {equip}, at {level} level{c}.", " and give me {n} exercises"),
    ("Can you build a {diff} {body} session{c} using {equip}?", " of {n} exercises"),
    ("{body} day! Using {equip}, {level} level{c}.", ", {n} exercises"),
    ("Plan a {body} workout for me: {level} level, {equip}{c}.", ", {n} moves"),
    ("{c}{body}, {equip}, {level} difficulty.", "{n} exercises for "),
    ("Need a {diff} workout for my {body} with {equip}{c}", ", {n} exercises please"),
    ("Make me a {body} workout. Equipment: {equip}. Difficulty: {level}.{c}", " Exercises: {n}."),
    ("{diff} {body} workout using {equip}{c}", ", {n} exercises"),
]
TEST_TEMPLATES = [  # never used in train/val: test also measures unseen wording
    ("What's a good {diff} {body} routine with {equip}?{c}", " Give me {n} exercises."),
    ("Hit my {body} with {equip}, {level} level{c}", ", {n} exercises"),
    ("I'm at the gym with {equip}. Can I get a {diff} {body} workout{c}?", " with {n} exercises"),
]


def all_combos():
    return [
        {"body_part": b, "equipment_group": g, "difficulty": d, "n_exercises": n}
        for b in BODY_WORDS for g in [*EQUIPMENT_GROUPS, "any"] for d in SETS_REPS for n in range(3, 7)
    ]


def user_message(rng, req, omit_count, templates):
    """-> (text, template index in `templates`)."""
    i = rng.randrange(len(templates))
    template, count_phrase = templates[i]
    n = req["n_exercises"]
    friendly, synonyms = BODY_WORDS[req["body_part"]]
    body = rng.choice(synonyms if rng.random() < SYNONYM_SHARE else friendly)
    c = "" if omit_count else count_phrase.format(n=NUMBER_WORDS[n] if rng.random() < 0.3 else n)
    text = template.format(body=body, equip=rng.choice(EQUIP_WORDS[req["equipment_group"]]),
                           diff=rng.choice(DIFF_WORDS[req["difficulty"]]),
                           level=rng.choice(LEVEL_WORDS[req["difficulty"]]), c=c)
    text = re.sub(r"\ba ([aeiou])", r"an \1", text)  # "a intermediate" -> "an intermediate"
    return text[0].upper() + text[1:], i


def assistant_answer(rng, req):
    names = matching_exercises(req)
    n = req["n_exercises"]
    if len(names) < n:
        obj = {"error": f"Only {len(names)} {req['body_part']} exercises match {req['equipment_group']} "
                        f"at {req['difficulty']} difficulty, but {n} were requested."}
    else:
        sets, reps = SETS_REPS[req["difficulty"]]
        obj = {"workout": [{"name": nm, "sets": sets, "reps": reps} for nm in rng.sample(names, n)]}
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def make_split(rng, combos, size, templates):
    """-> (rows, set of templates used)."""
    possible = [c for c in combos if is_feasible(c)]
    impossible = [c for c in combos if not is_feasible(c)]
    rows, used = [], set()
    for _ in range(size):
        req = rng.choice(impossible if rng.random() < IMPOSSIBLE_SHARE else possible)
        omit = req["n_exercises"] == 4 and rng.random() < OMIT_COUNT_SHARE  # a missing count means 4
        text, i = user_message(rng, req, omit, templates)
        used.add(templates[i])
        rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": text},
                {"role": "assistant", "content": assistant_answer(rng, req)},
            ],
            "meta": {**req, "feasible": is_feasible(req)},
        })
    return rows, used


def split_combos(rng):
    """Hold out TEST_COMBO_SHARE of possible and of impossible combos, separately, for test only."""
    held, kept = [], []
    for group in ([c for c in all_combos() if is_feasible(c)], [c for c in all_combos() if not is_feasible(c)]):
        group = group[:]
        rng.shuffle(group)
        k = round(len(group) * TEST_COMBO_SHARE)
        held += group[:k]
        kept += group[k:]
    return kept, held


def key(meta):
    return (meta["body_part"], meta["equipment_group"], meta["difficulty"], meta["n_exercises"])


if __name__ == "__main__":
    rng = random.Random(SEED)
    train_combos, test_combos = split_combos(rng)
    splits, templates_used = {}, {}
    for name, combos, templates in [("train", train_combos, TRAIN_TEMPLATES), ("val", train_combos, TRAIN_TEMPLATES),
                                    ("test", test_combos, TEST_TEMPLATES)]:
        splits[name], templates_used[name] = make_split(rng, combos, SIZES[name], templates)

    # self-checks: every answer passes the verifier; test combos and test templates never appear in train/val
    for name, rows in splits.items():
        for row in rows:
            r = verify(row["meta"], row["messages"][2]["content"])
            assert r["all_correct"], (name, row, r)
    seen = {key(r["meta"]) for s in ("train", "val") for r in splits[s]}
    assert not seen & {key(r["meta"]) for r in splits["test"]}, "test combo leaked into train/val"
    assert not (templates_used["train"] | templates_used["val"]) & set(TEST_TEMPLATES), "test template leaked"
    assert templates_used["test"] == set(TEST_TEMPLATES), "test split should use all 3 test-only templates"

    for name, rows in splits.items():
        with open(OUT / f"{name}.jsonl", "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)

    print(f"combos: {len(train_combos)} train/val, {len(test_combos)} test-only")
    print(f"templates: {len(TRAIN_TEMPLATES)} train/val, {len(TEST_TEMPLATES)} test-only")
    print("verifier: 100% all_correct\n")
    for name, rows in splits.items():
        share = sum(not r["meta"]["feasible"] for r in rows) / len(rows)
        print(f"{name:5} {len(rows):5} examples, impossible {share:.1%} -> data/processed/{name}.jsonl")
    for name in ("train", "test"):
        print(f"\n--- 5 random {name} examples ---")
        for row in random.Random(SEED + 1).sample(splits[name], 5):
            print("USER:     ", row["messages"][1]["content"])
            print("ASSISTANT:", row["messages"][2]["content"])
            print("META:     ", row["meta"], "\n")
