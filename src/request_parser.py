"""Free-text workout request -> the structured request verify() and is_feasible() need. Deterministic, no model.

    parse_request("Chest workout, medium difficulty, free weights, 5 exercises")
    # -> ({"body_part": "chest", "equipment_group": "free weights", "difficulty": "medium", "n_exercises": 5}, [])
    parse_request("biceps workout, easy, dumbbells")
    # -> ({"body_part": "arms", "equipment_group": "free weights", ...}, ["biceps → arms (...)", "dumbbells → ..."])

The words are make_dataset.py's request vocabulary (docs/spec.md "Request vocabulary"), plus SERVING_ALIASES:
everyday words real users type (biceps, dumbbells, ...). Aliases exist only here, never in the training data,
and each one used is reported back as a note. One regex finds every word, longest phrase first and as whole
words, so "forearm(s)" is not "arm(s)", "upper legs" is not "legs", "free weights" is not "weights",
"machines and cables" is not "machines". A count is 3-6, as digits or words; none means 4.
"""
import re

from make_dataset import BODY_WORDS, DIFF_WORDS, EQUIP_WORDS, LEVEL_WORDS, TRAIN_TEMPLATES
from spec import BODY_PARTS, EQUIPMENT_GROUPS
from verifier import DEFAULT_N_EXERCISES

MIN_N, MAX_N = 3, 6
# one..ten, not just three..six: "two exercises" must be refused, not silently read as the default 4
NUMBER_NAMES = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
PARTS = ["body_part", "equipment_group", "difficulty"]
EXAMPLE_REQUEST = "chest workout, medium difficulty, free weights, 5 exercises"
# Serving only (not in data-v1, not spec vocabulary). No "no equipment": the bodyweight group includes stability balls.
SERVING_ALIASES = {
    **dict.fromkeys(["biceps", "triceps", "bicep", "tricep", "arm"], ("body_part", "arms")),
    **dict.fromkeys(["quads", "quadriceps", "hamstrings", "leg"], ("body_part", "legs")),
    "lats": ("body_part", "back"), "shoulder": ("body_part", "shoulders"), "forearm": ("body_part", "forearms"),
    "calf": ("body_part", "calves"), "ab": ("body_part", "abs"),
    **dict.fromkeys(["dumbbell", "dumbbells", "barbell", "barbells", "kettlebell", "kettlebells"],
                    ("equipment_group", "free weights")),
    **dict.fromkeys(["machine", "cable", "cables"], ("equipment_group", "machines")),
    "body weight": ("equipment_group", "bodyweight"),
    **dict.fromkeys(["band", "resistance band"], ("equipment_group", "bands")),
}
# The sentence sent to the model when an alias was used: the first train template, the spec's own example shape.
CANONICAL_TEMPLATE = TRAIN_TEMPLATES[0]


def _vocabulary():
    """-> {phrase: (part, value)} for every spec request word."""
    vocab = {}
    entries = [("body_part", b, friendly + synonyms) for b, (friendly, synonyms) in BODY_WORDS.items()]
    entries += [("equipment_group", g, words) for g, words in EQUIP_WORDS.items()]
    entries += [("difficulty", d, DIFF_WORDS[d] + LEVEL_WORDS[d]) for d in DIFF_WORDS]
    entries += [("n_exercises", i, [w]) for i, w in enumerate(NUMBER_NAMES, 1)]
    for part, value, words in entries:
        for w in words:
            assert vocab.get(w, (part, value)) == (part, value), f"{w!r} means two things"
            vocab[w] = (part, value)
    return vocab


VOCAB = _vocabulary()
assert not VOCAB.keys() & SERVING_ALIASES.keys(), "an alias must not shadow a spec word"
WORDS = {**VOCAB, **SERVING_ALIASES}
ALLOWED = {part: {value: [w for w, pv in WORDS.items() if pv == (part, value)]
                  for value in dict.fromkeys(v for p, v in VOCAB.values() if p == part)} for part in PARTS}
ALLOWED["n_exercises"] = f"{MIN_N}-{MAX_N}, as digits or words ({', '.join(NUMBER_NAMES[MIN_N - 1:MAX_N])}); " \
                         f"left out = {DEFAULT_N_EXERCISES}"
# Longest phrase first, so at the same position "free weights" wins over a shorter phrase; inner spaces match any whitespace.
PATTERN = re.compile(r"\b(?:" + "|".join(r"\s+".join(map(re.escape, w.split()))
                                         for w in sorted(WORDS, key=len, reverse=True)) + r"|\d+)\b", re.IGNORECASE)


class RequestError(ValueError):
    """The request can't be read: parts missing, two different values for one part, or a count outside 3-6."""

    def __init__(self, missing, conflicts, bad_count):
        self.missing, self.conflicts, self.bad_count = missing, conflicts, bad_count
        problems = [f"missing {', '.join(missing)}"] if missing else []
        problems += [f"more than one {part}: {', '.join(map(str, values))}" for part, values in conflicts.items()]
        problems += [f"{bad_count} exercises: the count must be {MIN_N}-{MAX_N}"] if bad_count is not None else []
        super().__init__("; ".join(problems))

    def detail(self):
        """JSON-ready body for a 422 answer, with one valid request and the words each part accepts."""
        return {"message": f"{self}. Example of a valid request: {EXAMPLE_REQUEST!r}", "example": EXAMPLE_REQUEST,
                "missing": self.missing, "conflicts": self.conflicts, "bad_count": self.bad_count, "allowed": ALLOWED}


def _note(word, part, value):
    if part == "equipment_group":
        return f"{word} → {value} ({', '.join(EQUIPMENT_GROUPS[value])})"
    return f"{word} → {value} (catalog body part: {BODY_PARTS[value]})"


def parse_request(text):
    """-> ({"body_part", "equipment_group", "difficulty", "n_exercises"}, notes for the aliases used); raises RequestError."""
    found = {part: [] for part in [*PARTS, "n_exercises"]}
    notes = []
    for m in PATTERN.finditer(text):
        word = " ".join(m.group().lower().split())
        part, value = ("n_exercises", int(word)) if word.isdigit() else WORDS[word]
        if word in SERVING_ALIASES and _note(word, part, value) not in notes:
            notes.append(_note(word, part, value))
        if value not in found[part]:
            found[part].append(value)
    missing = [part for part in PARTS if not found[part]]
    conflicts = {part: values for part, values in found.items() if len(values) > 1}
    counts = found["n_exercises"]
    bad_count = counts[0] if len(counts) == 1 and not MIN_N <= counts[0] <= MAX_N else None
    if missing or conflicts or bad_count is not None:
        raise RequestError(missing, conflicts, bad_count)
    request = {**{part: found[part][0] for part in PARTS}, "n_exercises": counts[0] if counts else DEFAULT_N_EXERCISES}
    return request, notes


def canonical_text(request):
    """The request in spec words, as the model saw them in training: 'Arms workout, easy difficulty, free weights, 5 exercises'."""
    template, count_phrase = CANONICAL_TEMPLATE
    text = template.format(body=BODY_WORDS[request["body_part"]][0][0], level=LEVEL_WORDS[request["difficulty"]][0],
                           equip=EQUIP_WORDS[request["equipment_group"]][0],
                           c=count_phrase.format(n=request["n_exercises"]))
    return text[0].upper() + text[1:]


def model_input(text, request, notes):
    """The user text sent to the model: unchanged if only spec words were used (as in every evaluation),
    otherwise canonical_text(), so the model never sees a word it wasn't trained on."""
    return canonical_text(request) if notes else text
