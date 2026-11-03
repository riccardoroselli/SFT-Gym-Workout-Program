"""Break down the mistakes of one evaluated run. Reads eval/outputs/<run>.jsonl, prints tables, writes nothing.

    python src/analyze_errors.py --run base_fewshot3

1. first failed check per answer
2. sets/reps exactly right, per difficulty
3. names not in the catalog: case/spacing only, near-miss, invented (with examples)
4. body part / equipment right, using the closest catalog name for names not in the catalog
5. most common body-part mix-ups among exact catalog names
"""
import argparse
import collections
import difflib

from evaluate import OUTPUTS, read_jsonl
from spec import SETS_REPS
from verifier import CATALOG, _group_ok, _normalize, _parse, verify

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="name of a file in eval/outputs/, without .jsonl")
    analyze(parser.parse_args().run)
