"""Free-text workout request -> the structured request verify() and is_feasible() need. Deterministic, no model.

    parse_request("Chest workout, medium difficulty, free weights, 5 exercises")
    # -> {"body_part": "chest", "equipment_group": "free weights", "difficulty": "medium", "n_exercises": 5}

The words are make_dataset.py's request vocabulary (docs/spec.md "Request vocabulary"). One regex finds them,
longest phrase first and as whole words, so "forearms" is not "arms", "upper legs" is not "legs",
"free weights" is not "weights", "gym machines" is not "machines". A count is 3-6, as digits or words; none means 4.
"""
import re

from make_dataset import BODY_WORDS, DIFF_WORDS, EQUIP_WORDS, LEVEL_WORDS
from verifier import DEFAULT_N_EXERCISES

MIN_N, MAX_N = 3, 6
# one..ten, not just three..six: "two exercises" must be refused, not silently read as the default 4
NUMBER_NAMES = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
PARTS = ["body_part", "equipment_group", "difficulty"]


def _vocabulary():
    """-> {phrase: (part, value)} for every request word."""
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
ALLOWED = {part: {value: [w for w, pv in VOCAB.items() if pv == (part, value)]
                  for value in dict.fromkeys(v for p, v in VOCAB.values() if p == part)} for part in PARTS}
ALLOWED["n_exercises"] = f"{MIN_N}-{MAX_N}, as digits or words ({', '.join(NUMBER_NAMES[MIN_N - 1:MAX_N])}); " \
                         f"left out = {DEFAULT_N_EXERCISES}"
# Longest phrase first, so at the same position "free weights" wins over a shorter phrase; inner spaces match any whitespace.
PATTERN = re.compile(r"\b(?:" + "|".join(r"\s+".join(map(re.escape, w.split()))
                                         for w in sorted(VOCAB, key=len, reverse=True)) + r"|\d+)\b", re.IGNORECASE)


class RequestError(ValueError):
    """The request can't be read: parts missing, two different values for one part, or a count outside 3-6."""

    def __init__(self, missing, conflicts, bad_count):
        self.missing, self.conflicts, self.bad_count = missing, conflicts, bad_count
        problems = [f"missing {', '.join(missing)}"] if missing else []
        problems += [f"more than one {part}: {', '.join(map(str, values))}" for part, values in conflicts.items()]
        problems += [f"{bad_count} exercises: the count must be {MIN_N}-{MAX_N}"] if bad_count is not None else []
        super().__init__("; ".join(problems))

    def detail(self):
        """JSON-ready body for a 422 answer, with the words each part accepts."""
        return {"message": str(self), "missing": self.missing, "conflicts": self.conflicts,
                "bad_count": self.bad_count, "allowed": ALLOWED}


def parse_request(text):
    """-> {"body_part", "equipment_group", "difficulty", "n_exercises"}; raises RequestError."""
    found = {part: [] for part in [*PARTS, "n_exercises"]}
    for m in PATTERN.finditer(text):
        word = " ".join(m.group().lower().split())
        part, value = ("n_exercises", int(word)) if word.isdigit() else VOCAB[word]
        if value not in found[part]:
            found[part].append(value)
    missing = [part for part in PARTS if not found[part]]
    conflicts = {part: values for part, values in found.items() if len(values) > 1}
    counts = found["n_exercises"]
    bad_count = counts[0] if len(counts) == 1 and not MIN_N <= counts[0] <= MAX_N else None
    if missing or conflicts or bad_count is not None:
        raise RequestError(missing, conflicts, bad_count)
    return {**{part: found[part][0] for part in PARTS}, "n_exercises": counts[0] if counts else DEFAULT_N_EXERCISES}
