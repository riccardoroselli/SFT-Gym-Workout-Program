"""Score a model answer against the task contract in spec.md.

    from verifier import verify, is_feasible
    request = {"body_part": "chest", "equipment_group": "free weights", "difficulty": "medium", "n_exercises": 4}
    verify(request, model_output_text)  # -> dict of checks + "reward" in [0, 1]
"""
import json
import re
from pathlib import Path

from spec import ADVANCED_ALLOWED, BODY_PARTS, EQUIPMENT_GROUPS, GROUP_OF, SETS_REPS

ROOT = Path(__file__).resolve().parent.parent
CATALOG = {e["name"]: e for e in json.loads((ROOT / "data/processed/catalog.json").read_text())}
DEFAULT_N_EXERCISES = 4

FORMAT_WEIGHT = 0.10  # share of the reward for strict_format; the rest is content
CONTENT_WEIGHTS = {   # for a workout answer to a feasible request; sums to 1
    "names_exist": 0.30,  # an invented exercise is the worst mistake
    "body_part_ok": 0.15,
    "equipment_ok": 0.15,
    "count_ok": 0.10,
    "no_duplicates": 0.10,
    "sets_reps_ok": 0.10,
    "advanced_ok": 0.10,
}
WORKOUT_CHECKS = list(CONTENT_WEIGHTS)
FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _normalize(request):
    """Request dict -> (catalog body_part, equipment group, difficulty, n). Bad values are our bug: raise."""
    if request["body_part"] not in BODY_PARTS:
        raise ValueError(f"unknown body_part {request['body_part']!r}")
    group = request["equipment_group"]
    if group != "any" and group not in EQUIPMENT_GROUPS:
        raise ValueError(f"unknown equipment_group {group!r}")
    if request["difficulty"] not in SETS_REPS:
        raise ValueError(f"unknown difficulty {request['difficulty']!r}")
    n = request.get("n_exercises") or DEFAULT_N_EXERCISES
    return BODY_PARTS[request["body_part"]], group, request["difficulty"], n


def _group_ok(exercise, group):
    return group == "any" or GROUP_OF[exercise["equipment"]] == group


def matching_exercises(request):
    """Catalog names that fit body part + equipment group (+ no advanced unless the difficulty allows it)."""
    body, group, difficulty, _ = _normalize(request)
    return [name for name, e in CATALOG.items()
            if e["body_part"] == body and _group_ok(e, group) and (difficulty in ADVANCED_ALLOWED or not e["advanced"])]


def is_feasible(request):
    """True if the catalog has at least n matching exercises."""
    return len(matching_exercises(request)) >= _normalize(request)[3]


def _parse(text):
    """-> (object or None, strict_format). Lenient: strips ``` fences and text around one JSON object."""
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict):
            return obj, True
    except json.JSONDecodeError:
        pass
    fenced = FENCE.search(text)
    body = fenced.group(1) if fenced else text
    start = body.find("{")
    if start < 0:
        return None, False
    try:
        obj, _ = json.JSONDecoder().raw_decode(body, start)
    except json.JSONDecodeError:
        return None, False
    return (obj if isinstance(obj, dict) else None), False


def _check_workout(items, body, group, difficulty, n):
    """Workout checks; fractions are over all listed items, an empty list fails everything."""
    if not items:
        return {"names_exist": 0.0, "body_part_ok": 0.0, "equipment_ok": 0.0,
                "count_ok": False, "no_duplicates": False, "sets_reps_ok": False, "advanced_ok": False}
    names = [it.get("name") if isinstance(it, dict) else None for it in items]
    found = [CATALOG.get(nm) if isinstance(nm, str) else None for nm in names]
    sets, reps = SETS_REPS[difficulty]
    return {
        "names_exist": sum(e is not None for e in found) / len(items),
        "body_part_ok": sum(e is not None and e["body_part"] == body for e in found) / len(items),
        "equipment_ok": sum(e is not None and _group_ok(e, group) for e in found) / len(items),
        "count_ok": len(items) == n,
        "no_duplicates": len(set(map(str, names))) == len(names),
        "sets_reps_ok": all(isinstance(it, dict) and type(it.get("sets")) is int and it["sets"] == sets
                            and it.get("reps") == reps for it in items),
        "advanced_ok": difficulty in ADVANCED_ALLOWED or not any(e is not None and e["advanced"] for e in found),
    }


def verify(request, output_text):
    body, group, difficulty, n = _normalize(request)
    feasible = is_feasible(request)
    obj, strict = _parse(output_text)

    is_workout = obj is not None and isinstance(obj.get("workout"), list) and "error" not in obj
    is_error = obj is not None and isinstance(obj.get("error"), str) and "workout" not in obj
    result = {"parse_ok": is_workout or is_error, "strict_format": strict}
    result.update(_check_workout(obj["workout"], body, group, difficulty, n) if is_workout
                  else dict.fromkeys(WORKOUT_CHECKS))  # None: not a workout answer
    result["error_correct"] = None if feasible else (is_error and obj["error"].strip() != "")

    if feasible:
        workout_perfect = is_workout and all(result[k] in (True, 1.0) for k in WORKOUT_CHECKS)
        content = sum(w * float(result[k]) for k, w in CONTENT_WEIGHTS.items()) if is_workout else 0.0
    else:
        workout_perfect = False
        content = float(result["error_correct"])
    result["all_correct"] = strict and (workout_perfect if feasible else result["error_correct"])
    result["reward"] = 0.0 if not result["parse_ok"] else FORMAT_WEIGHT * strict + (1 - FORMAT_WEIGHT) * content
    return result
