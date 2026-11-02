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

Exercises with any other equipment (rope, tire, stationary bike, …: 12 in the catalog) are never used.

## 2. The answer
```json
{"workout": [{"name": "...", "sets": 3, "reps": "8-12"}]}
```
- `name` is copied **exactly** from `catalog.json`, and the exercise must match the body part and equipment group.
- No exercise appears twice. The list has exactly the requested number of exercises.
- Sets and reps depend only on the workout difficulty: **easy** 2 × "12-15", **medium** 3 × "8-12", **hard** 4 × "5-8".

## 3. Difficulty rule
Difficulty is decided by **sets and reps** (section 2), not by the exercises. The only exercise rule:
- **easy** workout → no exercise with `"advanced": true` in `catalog.json` (skills a beginner can't do at any rep count: pull-ups, dips, planches, muscle-ups, pistols, Olympic lifts, …).
- **medium** and **hard** workouts → any exercise that matches body part and equipment.

## 4. Impossible requests
If not enough exercises match (body part + equipment, and no advanced ones for easy), answer with an error instead of inventing or repeating exercises:
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
