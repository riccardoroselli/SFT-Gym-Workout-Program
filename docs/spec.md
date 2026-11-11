# Task contract

The model reads a short workout request and answers with **only JSON**: a workout, or an error.
Every exercise comes from `data/processed/catalog.json`.

## 1. The request
A sentence like *"chest workout, medium difficulty, free weights, 5 exercises"*, with four parts:

| Part | Allowed values |
|---|---|
| **Body part** | `chest`, `back`, `shoulders`, `arms` (= upper arms), `forearms` (= lower arms), `legs` (= upper legs), `calves` (= lower legs), `abs`/`core` (= waist) |
| **Equipment** | **free weights**: dumbbell, barbell, ez barbell, olympic barbell, kettlebell, trap bar, weighted, medicine ball · **machines**: cable, leverage machine, smith machine, sled machine, assisted · **bodyweight**: body weight, stability ball, bosu ball, roller, wheel roller · **bands**: band, resistance band · **any**: all four groups |
| **Difficulty** | `easy`, `medium`, `hard` |
| **Number of exercises** | 3–6; if missing, use **4** |

Exercises with any other equipment (rope, tire, hammer, …: 8 exercises) are never used.

### Request vocabulary
Every word `src/make_dataset.py` uses in requests, and what it means. Body parts use the friendly name ~80% of the time and a synonym ~20%.

| Part | User says | Maps to |
|---|---|---|
| Body part | `chest` · synonyms: `pecs` | `chest` |
| Body part | `back` · synonyms: `lats and upper back` | `back` |
| Body part | `shoulders` · synonyms: `delts` | `shoulders` |
| Body part | `arms` · synonyms: `upper arms`, `biceps and triceps` | `upper arms` |
| Body part | `forearms` · synonyms: `lower arms` | `lower arms` |
| Body part | `legs` · synonyms: `upper legs`, `thighs` | `upper legs` |
| Body part | `calves` · synonyms: `lower legs` | `lower legs` |
| Body part | `abs`, `core` · synonyms: `abdominals`, `midsection`, `waist` | `waist` |
| Equipment | `free weights`, `weights` | **free weights** group |
| Equipment | `machines`, `gym machines`, `machines and cables` | **machines** group |
| Equipment | `bodyweight`, `bodyweight exercises`, `calisthenics` | **bodyweight** group |
| Equipment | `bands`, `resistance bands`, `elastic bands` | **bands** group |
| Equipment | `any equipment`, `whatever equipment`, `anything in the gym` | all four groups |
| Difficulty | `easy`, `beginner`, `light` | `easy` (2 × "12-15") |
| Difficulty | `medium`, `intermediate`, `moderate` | `medium` (3 × "8-12") |
| Difficulty | `hard`, `advanced`, `intense` | `hard` (4 × "5-8") |
| Count | `3`–`6` or `three`–`six`; left out | that many exercises; **4** |

## 2. The answer
```json
{"workout": [{"name": "...", "sets": 3, "reps": "8-12"}]}
```
- `name` is copied **exactly** from `catalog.json`, and the exercise must match the body part and equipment group.
- No exercise appears twice. The list has exactly the requested number of exercises.
- **The same exercise** = two names that match after lowercasing and removing "v. N" (any number), "(male)" and "(female)": `dumbbell front raise` and `dumbbell front raise v. 2` are one exercise and can't both be listed. "on exercise ball" is a different exercise: `dumbbell incline fly` and `dumbbell incline fly on exercise ball` can be listed together.
- Sets and reps depend only on the workout difficulty: **easy** 2 × "12-15", **medium** 3 × "8-12", **hard** 4 × "5-8".

## 3. Difficulty rule
Difficulty is decided by **sets and reps** (section 2), not by the exercises. The only exercise rule:
- **easy** and **medium** workouts → no exercise with `"advanced": true` in `catalog.json` (skills a beginner can't do at any rep count: pull-ups, dips, planches, muscle-ups, pistols, Olympic lifts, …).
- **hard** workouts → any exercise that matches body part and equipment, advanced ones included.

## 4. Impossible requests
If not enough different exercises match (body part + equipment, and no advanced ones unless the workout is hard; variants of the same exercise count once), answer with an error instead of inventing or repeating exercises:
```json
{"error": "..."}
```
The message says in plain words what was asked and how many exercises matched.

## Examples
**Valid**: *"chest workout, medium difficulty, free weights"* (no count → 4)
```json
{"workout": [
  {"name": "barbell bench press", "sets": 3, "reps": "8-12"},
  {"name": "dumbbell bench press", "sets": 3, "reps": "8-12"},
  {"name": "barbell incline bench press", "sets": 3, "reps": "8-12"},
  {"name": "barbell decline bench press", "sets": 3, "reps": "8-12"}
]}
```
**Error**: *"forearms workout, easy, bands, 3 exercises"* (the catalog has only 2 forearm exercises with bands)
```json
{"error": "Only 2 forearms exercises use bands, but 3 were requested."}
```

## Known limitations
- **Impossible requests come from only 5 body part × equipment pairs:** forearms + bands (2 exercises), forearms + bodyweight (2), shoulders + bodyweight (1), and calves + bands and chest + bands (3 each, so they fail from 4 exercises up). The model can learn these pairs by heart instead of learning to count matching exercises, so its error behaviour may not transfer if the catalog changes.
- **data-v1 predates the "same exercise" rule:** 19 train answers (0 val, 0 test) list a variant pair, e.g. `jump squat` with `jump squat v. 2`, so they fail `no_duplicates` under today's verifier. `make_dataset.py`'s self-check accepts them (`verify(..., variants_distinct=True)`) so the frozen files still regenerate byte-identical. Feasibility is unchanged: no request combination flips.
