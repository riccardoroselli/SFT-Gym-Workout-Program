"""Build data/processed/catalog.json from data/raw/exercises.json. This is the source of truth for the catalog.

    python src/build_catalog.py
"""
import json
from pathlib import Path

import pandas as pd

from spec import BODY_PARTS, GROUP_OF

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/exercises.json"
CATALOG = ROOT / "data/processed/catalog.json"
ADVANCED = ROOT / "data/processed/advanced_ids.json"
COLUMNS = ["id", "name", "body_part", "target", "equipment", "secondary_muscles"]


def build():
    df = pd.read_json(RAW, dtype={"id": str})  # keep "0001", not 1
    names = df["name"].str.lower()
    df = df[~names.str.contains("stretch", regex=False)]      # warm-up moves, not working sets
    df = df[df["body_part"].isin(BODY_PARTS.values())]        # drops neck and cardio: not requestable
    df = df[~df["name"].str.lower().duplicated(keep="first")]  # copies with identical labels
    df = df[df["equipment"].isin(GROUP_OF)]                    # equipment outside every group
    df = df[COLUMNS].reset_index(drop=True)

    advanced = set(json.loads(ADVANCED.read_text()))
    unknown = advanced - set(df["id"])
    assert not unknown, f"{len(unknown)} advanced ids not in the catalog, e.g. {sorted(unknown)[:5]}"
    df["advanced"] = df["id"].isin(advanced)
    return df


if __name__ == "__main__":
    df = build()
    dups = df["name"].str.lower().duplicated().sum()
    stretches = df["name"].str.lower().str.contains("stretch", regex=False).sum()
    assert dups == 0 and stretches == 0, (dups, stretches)

    df.to_json(CATALOG, orient="records", indent=2, force_ascii=False)
    print(f"saved -> {CATALOG.relative_to(ROOT)}")
    print(f"total exercises: {len(df)}\n")
    print(df["body_part"].value_counts().to_string(), "\n")
    print(f"duplicate names: {dups}")
    print(f"stretches: {stretches}")
    print(f"advanced: {df['advanced'].sum()}")
