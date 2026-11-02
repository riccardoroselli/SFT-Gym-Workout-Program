"""Review the advanced flags (see spec.md section 3).

Workout difficulty comes from sets/reps. The only exercise rule: advanced skills
(moves a beginner can't do at any rep count) are allowed only in hard workouts.
Claude Code read the exercises batch by batch and listed the advanced ones, following
RUBRIC, in data/processed/advanced_ids.json. build_catalog.py applies that list.

    python src/label_advanced.py batches   # print the exercises to review, 20 per batch, as JSON
    python src/label_advanced.py report    # exercises available per request, and the advanced list
"""
import json
import sys
from pathlib import Path

import pandas as pd

from spec import ADVANCED_ALLOWED, GROUP_OF

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data/processed/catalog.json"
BATCH_SIZE = 20

RUBRIC = """advanced = a beginner can't do it at any rep count, or can't do it safely:
calisthenics skills (planche, lever, muscle-up, handstand, L-sit), strict pull-ups/chin-ups/bar dips,
straight-leg hanging raises, one-arm/one-leg skills (pistol, one-arm push-up), explosive push-ups,
Olympic-style lifts (clean, snatch, jerk, get-up), Nordic-style leg curls.
Heavy but learnable lifts (squat, bench, deadlift, lunges) are NOT advanced."""


def print_batches(df):
    cols = ["id", "name", "body_part", "target", "equipment"]
    for start in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[start:start + BATCH_SIZE][cols].to_dict(orient="records")
        print(f"# batch {start // BATCH_SIZE + 1}")
        print(json.dumps(batch, ensure_ascii=False))


def report(df):
    # how many exercises each request can draw from: hard uses all, easy/medium skip advanced
    assert ADVANCED_ALLOWED == {"hard"}, "update the column names below"
    df["group"] = df["equipment"].map(GROUP_OF)
    table = df.groupby(["body_part", "group"]).agg(hard=("id", "size"), easy_medium=("advanced", lambda a: (~a).sum()))
    print(table.to_string(), "\n")
    print(f"advanced exercises ({df['advanced'].sum()}):")
    print(df[df["advanced"]][["name", "body_part", "equipment"]].to_string(index=False))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    catalog = pd.read_json(CATALOG, dtype={"id": str})
    if mode == "batches":
        print_batches(catalog)
    elif mode == "report":
        report(catalog)
    else:
        sys.exit(__doc__)
