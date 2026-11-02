"""Flag the advanced exercises in data/processed/catalog.json (see spec.md section 3).

Workout difficulty comes from sets/reps. The only exercise rule: advanced skills
(moves a beginner can't do at any rep count) are kept out of easy workouts.
Claude Code read the exercises batch by batch and listed the advanced ones, following
RUBRIC, in data/processed/advanced_ids.json.

    python src/label_advanced.py batches   # print the exercises to review, 20 per batch, as JSON
    python src/label_advanced.py apply     # set "advanced" in catalog.json and print reports
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data/processed/catalog.json"
ADVANCED = ROOT / "data/processed/advanced_ids.json"
BATCH_SIZE = 20

RUBRIC = """advanced = a beginner can't do it at any rep count, or can't do it safely:
calisthenics skills (planche, lever, muscle-up, handstand, L-sit), strict pull-ups/chin-ups/bar dips,
straight-leg hanging raises, one-arm/one-leg skills (pistol, one-arm push-up), explosive push-ups,
Olympic-style lifts (clean, snatch, jerk, get-up), Nordic-style leg curls.
Heavy but learnable lifts (squat, bench, deadlift, lunges) are NOT advanced."""

# Same groups as spec.md section 1.
EQUIPMENT_GROUPS = {
    "free weights": ["dumbbell", "barbell", "ez barbell", "olympic barbell", "kettlebell", "trap bar", "weighted", "medicine ball"],
    "machines": ["cable", "leverage machine", "smith machine", "sled machine", "assisted"],
    "bodyweight": ["body weight", "stability ball", "bosu ball", "roller", "wheel roller"],
    "bands": ["band", "resistance band"],
}
GROUP_OF = {eq: group for group, eqs in EQUIPMENT_GROUPS.items() for eq in eqs}


def load_catalog():
    """Catalog without exercises whose equipment is in no group (rope, tire, bikes, ...)."""
    df = pd.read_json(CATALOG, dtype={"id": str})
    df = df.drop(columns=["difficulty", "advanced"], errors="ignore")  # old labels from earlier runs
    return df[df["equipment"].isin(GROUP_OF)].reset_index(drop=True)


def print_batches(df):
    cols = ["id", "name", "body_part", "target", "equipment"]
    for start in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[start:start + BATCH_SIZE][cols].to_dict(orient="records")
        print(f"# batch {start // BATCH_SIZE + 1}")
        print(json.dumps(batch, ensure_ascii=False))


def apply_labels(df):
    advanced = set(json.loads(ADVANCED.read_text()))
    unknown = advanced - set(df["id"])
    assert not unknown, f"{len(unknown)} advanced ids not in the catalog, e.g. {sorted(unknown)[:5]}"

    df["advanced"] = df["id"].isin(advanced)
    df.to_json(CATALOG, orient="records", indent=2, force_ascii=False)
    print(f"saved {len(df)} exercises, {df['advanced'].sum()} advanced -> {CATALOG.relative_to(ROOT)}\n")

    # how many exercises each request can draw from: medium/hard use all, easy skips advanced
    df["group"] = df["equipment"].map(GROUP_OF)
    table = df.groupby(["body_part", "group"]).agg(medium_hard=("id", "size"), easy=("advanced", lambda a: (~a).sum()))
    print(table.to_string(), "\n")
    print("advanced exercises:")
    print(df[df["advanced"]][["name", "body_part", "equipment"]].to_string(index=False))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "batches":
        print_batches(load_catalog())
    elif mode == "apply":
        apply_labels(load_catalog())
    else:
        sys.exit(__doc__)
