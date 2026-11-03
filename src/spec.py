"""Task contract values from docs/spec.md, shared by every script. Change them here, nowhere else."""

# Section 1: equipment groups. Exercises with any other equipment are excluded.
EQUIPMENT_GROUPS = {
    "free weights": ["dumbbell", "barbell", "ez barbell", "olympic barbell", "kettlebell", "trap bar", "weighted", "medicine ball"],
    "machines": ["cable", "leverage machine", "smith machine", "sled machine", "assisted"],
    "bodyweight": ["body weight", "stability ball", "bosu ball", "roller", "wheel roller"],
    "bands": ["band", "resistance band"],
}
GROUP_OF = {eq: group for group, eqs in EQUIPMENT_GROUPS.items() for eq in eqs}

# Section 1: friendly body part name in the request -> body_part in the catalog.
BODY_PARTS = {
    "chest": "chest",
    "back": "back",
    "shoulders": "shoulders",
    "arms": "upper arms",
    "forearms": "lower arms",
    "legs": "upper legs",
    "calves": "lower legs",
    "abs": "waist",
    "core": "waist",
}

# Section 2: workout difficulty -> (sets, reps).
SETS_REPS = {
    "easy": (2, "12-15"),
    "medium": (3, "8-12"),
    "hard": (4, "5-8"),
}

# Section 3: difficulties whose workouts may include exercises with "advanced": true.
ADVANCED_ALLOWED = {"hard"}
