"""Break down the mistakes of one evaluated run. Reads eval/outputs/<run>.jsonl, prints tables, writes nothing.

    python src/analyze_errors.py --run base_fewshot3

1. first failed check per answer
2. sets/reps exactly right, per difficulty
3. names not in the catalog: case/spacing only, near-miss, invented (with examples)
4. body part / equipment right, using the closest catalog name for names not in the catalog
5. most common body-part mix-ups among exact catalog names
6. duplicates: exact repeats, and near-duplicates (equal once "v. 2", "(male)", ... are removed)
7. variety: valid names used per (body part, equipment) pair, against the catalog pool
8. impossible requests: refused or answered with a workout, per combo
9. all_correct split by difficulty, equipment group, number of exercises, test template
"""
import argparse
import collections
import difflib
import re

from evaluate import OUTPUTS, read_jsonl
from make_dataset import TEST_TEMPLATES
from spec import SETS_REPS
from verifier import CATALOG, _group_ok, _normalize, _parse, matching_exercises, verify

NEAR_MISS = 0.85  # difflib ratio; names just below it are often the right exercise with extra words
# (label, verifier field) in the order the checks are tried
CHECKS = [("parse", "parse_ok"), ("names", "names_exist"), ("body part", "body_part_ok"),
          ("equipment", "equipment_ok"), ("count", "count_ok"), ("duplicates", "no_duplicates"),
          ("sets/reps", "sets_reps_ok"), ("advanced", "advanced_ok")]


def first_failed(meta, result):
    if result["all_correct"]:
        return "passed"
    if not meta["feasible"]:  # the verifier scores impossible requests only on the error answer
        return "error handling"
    return next((label for label, k in CHECKS if result[k] not in (True, 1.0)), "strict format only")


def _norm(name):
    return " ".join(name.lower().split())


CATALOG_NORM = {_norm(n): n for n in CATALOG}


def closest(name):
    """-> (kind, closest catalog name, similarity) for a name that is not exactly in the catalog."""
    if _norm(name) in CATALOG_NORM:
        return "case/spacing", CATALOG_NORM[_norm(name)], 1.0
    ratio = lambda c: difflib.SequenceMatcher(None, _norm(name), c.lower()).ratio()
    best = max(CATALOG, key=ratio)
    return ("near-miss" if ratio(best) >= NEAR_MISS else "invented"), best, ratio(best)


def items_of(output):
    """The listed exercises of a workout answer (lenient parse, like the verifier), else []."""
    obj, _ = _parse(output)
    items = obj.get("workout") if obj else None
    return [it for it in items if isinstance(it, dict)] if isinstance(items, list) else []


SUFFIX = re.compile(r"\s*(v\.\s*\d+|\((?:male|female)\)|on exercise ball)\s*", re.I)
TEMPLATE_PREFIXES = [t.split("{")[0] for t, _ in TEST_TEMPLATES]  # "What's a good ", "Hit my ", ...


def strip_variant(name):
    """'dumbbell rear fly v. 2' -> 'dumbbell rear fly'; removes every variant suffix, lowercases."""
    return " ".join(SUFFIX.sub(" ", name).lower().split())


def duplicates(names):
    """-> (extra exact copies, True if two different names are equal after strip_variant)."""
    distinct = set(names)
    return len(names) - len(distinct), len({strip_variant(n) for n in distinct}) < len(distinct)


def template_of(request):
    return next((p.strip() + " ..." for p in TEMPLATE_PREFIXES if request.startswith(p)), "other")


def analyze(run):
    rows = read_jsonl(OUTPUTS / f"{run}.jsonl")
    first = collections.Counter()
    sr_total, sr_ok = collections.Counter(), collections.Counter()
    kinds = collections.defaultdict(list)  # kind -> [(name, closest, ratio, meta)]
    exact, mixups, cache, n_items = [], collections.Counter(), {}, 0

    for row in rows:
        meta = row["meta"]
        first[first_failed(meta, verify(meta, row["output"]))] += 1
        body, _, difficulty, _ = _normalize(meta)
        for it in items_of(row["output"]):
            n_items += 1
            sr_total[difficulty] += 1
            sr_ok[difficulty] += (it.get("sets"), it.get("reps")) == SETS_REPS[difficulty]
            name = it.get("name")
            if not isinstance(name, str):
                continue
            if name in CATALOG:
                exact.append((name, name, 1.0, meta))
                if CATALOG[name]["body_part"] != body:
                    mixups[(body, CATALOG[name]["body_part"])] += 1
            else:
                cache.setdefault(name, closest(name))
                kind, best, ratio = cache[name]
                kinds[kind].append((name, best, ratio, meta))

    print(f"== {run}: {len(rows)} answers, {sum(not r['meta']['feasible'] for r in rows)} impossible requests\n")
    print("1. first failed check (impossible requests count as error handling)")
    for label in [l for l, _ in CHECKS] + ["error handling", "strict format only", "passed"]:
        print(f"   {label:18} {first[label]}")

    print("\n2. sets/reps exactly right, per exercise")
    for d in SETS_REPS:
        if sr_total[d]:
            print(f"   {d:6} {sr_ok[d]}/{sr_total[d]} ({sr_ok[d] / sr_total[d]:.0%})  expected {SETS_REPS[d]}")

    print(f"\n3. listed exercises: {n_items}, exact catalog names: {len(exact)}")
    print(f"   names not in the catalog (near-miss = difflib ratio >= {NEAR_MISS}):")
    for kind in ["case/spacing", "near-miss", "invented"]:
        group = kinds[kind]
        print(f"   {kind:13} {len(group)} occurrences, {len({g[0] for g in group})} unique")
        for name, best, ratio in list(dict.fromkeys((g[0], g[1], g[2]) for g in group))[:2]:
            print(f"      {name!r} -> {best!r} ({ratio:.2f})")

    print("\n4. body part / equipment right (closest catalog name for names not in the catalog)")
    for label, group in [("exact", exact), ("case/spacing", kinds["case/spacing"]),
                         ("near-miss", kinds["near-miss"]), ("invented", kinds["invented"])]:
        if not group:
            continue
        bp = sum(CATALOG[best]["body_part"] == _normalize(m)[0] for _, best, _, m in group)
        eq = sum(_group_ok(CATALOG[best], _normalize(m)[1]) for _, best, _, m in group)
        print(f"   {label:13} body part {bp}/{len(group)} ({bp / len(group):.0%}), "
              f"equipment {eq}/{len(group)} ({eq / len(group):.0%})")

    print("\n5. body-part mix-ups among exact catalog names (requested -> given)")
    for (asked, given), count in mixups.most_common(5):
        print(f"   {asked} -> {given}: {count}")

    results = [verify(r["meta"], r["output"]) for r in rows]
    names_of = [[it["name"] for it in items_of(r["output"]) if isinstance(it.get("name"), str)] for r in rows]
    workouts = [(r, ns) for r, ns in zip(rows, names_of) if ns]

    print("\n6. duplicates (workout answers only)")
    dups = [duplicates(ns) for _, ns in workouts]
    with_exact = [extra for extra, _ in dups if extra]
    repeated = collections.Counter()
    for _, ns in workouts:
        repeated.update({n: k - 1 for n, k in collections.Counter(ns).items() if k > 1})
    share = lambda k: f"{k}/{len(workouts)} ({k / len(workouts):.0%})" if workouts else "0/0"
    print(f"   with an exact duplicate: {share(len(with_exact))}"
          + (f", {sum(with_exact) / len(with_exact):.2f} extra copies on average" if with_exact else ""))
    for name, k in repeated.most_common(5):
        print(f"      {k:3} extra  {name}")
    print(f"   with a near-duplicate:   {share(sum(near for _, near in dups))}")
    print(f"   with either:             {share(sum(bool(extra) or near for extra, near in dups))}")

    print("\n7. variety: distinct valid names used per (body part, equipment) pair, feasible requests")
    used = collections.defaultdict(set)
    for r, ns in workouts:
        if r["meta"]["feasible"]:
            used[(r["meta"]["body_part"], r["meta"]["equipment_group"])].update(ns)
    pools = {(b, g): set(matching_exercises({"body_part": b, "equipment_group": g, "difficulty": "hard"}))
             for b, g in used}  # hard pool = the biggest one (advanced allowed)
    cover = {pair: (len(used[pair] & pools[pair]), len(pools[pair])) for pair in used}
    if cover:
        hit, pool = sum(h for h, _ in cover.values()), sum(p for _, p in cover.values())
        print(f"   overall: {hit} of {pool} (pair, exercise) slots used ({hit / pool:.0%}), {len(cover)} pairs")
        for (b, g), (h, p) in sorted(cover.items(), key=lambda kv: (kv[1][0] / kv[1][1], -kv[1][1]))[:3]:
            print(f"   worst: {b} / {g}: {h} of {p} ({h / p:.0%})")

    print("\n8. impossible requests: refused with {\"error\": ...} or answered with a workout")
    combos = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        m = r["meta"]
        if not m["feasible"]:
            obj, _ = _parse(r["output"])
            combos[(m["body_part"], m["equipment_group"], m["difficulty"], m["n_exercises"])][
                0 if isinstance(obj, dict) and "error" in obj else 1] += 1
    print(f"   refused {sum(c[0] for c in combos.values())}, workout {sum(c[1] for c in combos.values())}")
    for (b, g, d, n), (ref, wk) in sorted(combos.items()):
        pool = len(matching_exercises({"body_part": b, "equipment_group": g, "difficulty": d}))
        print(f"   {b}/{g}/{d}, {n} asked, pool {pool}: refused {ref}, workout {wk}")

    print("\n9. all_correct splits")
    for label, key in [("difficulty", lambda r: r["meta"]["difficulty"]),
                       ("equipment", lambda r: r["meta"]["equipment_group"]),
                       ("n exercises", lambda r: r["meta"]["n_exercises"]),
                       ("template", lambda r: template_of(r["request"]))]:
        groups = collections.defaultdict(list)
        for r, res in zip(rows, results):
            groups[str(key(r))].append(res["all_correct"])
        print(f"   by {label}: " + ", ".join(f"{k} {sum(v) / len(v):.2f} (n={len(v)})" for k, v in sorted(groups.items())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="name of a file in eval/outputs/, without .jsonl")
    analyze(parser.parse_args().run)
