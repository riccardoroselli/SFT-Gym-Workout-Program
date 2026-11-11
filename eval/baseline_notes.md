# Baseline notes: Qwen3.5-2B before fine-tuning

Test set: 300 requests (unseen combinations and unseen phrasings), 16 of them impossible. Greedy decoding, thinking off.
`base_fewshot3` gets train rows 2, 3, 15 as earlier chat turns: a medium workout, a hard workout and an impossible easy request (error answer). The 3 shots cover medium, hard and impossible, but **no easy workout**.
Breakdowns below come from `python src/analyze_errors.py --run base_fewshot3`.

| run | n | parse_ok | strict_format | names_exist | body_part_ok | equipment_ok | count_ok | no_duplicates | sets_reps_ok | advanced_ok | error_correct | all_correct | reward | has_think | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_zeroshot | 300 | 0.0 | 0.0 | - | - | - | - | - | - | - | 0.0 | 0.0 | 0.0 | 0.0 | 0.14 |
| base_fewshot3 | 300 | 0.9967 | 0.9967 | 0.1611 | 0.0915 | 0.1451 | 0.8829 | 0.9431 | 0.0167 | 1.0 | 0.0 | 0.0 | 0.4147 | 0.0 | 0.0 |

First failed check in `base_fewshot3`, in the order parse, names, body part, equipment, count, duplicates, sets/reps, advanced, error handling. **Impossible requests are counted under error handling**, whatever else is wrong, because the error answer is the only thing the verifier scores on them:
names 279 · error handling 16 · body part 4 · parse 1 · passed 0.

## Observations

1. **Zero-shot never matches the shape.** None of the 300 answers is `{"workout": [...]}` or `{"error": ...}`. The model invents its own JSON (`workout_plan`, `focus`, `notes`, `total_exercises`), uses Title Case names, sometimes writes invalid JSON (`"reps": 8-10`), and 14% of answers are cut off at 512 tokens. Reward 0.

2. **Names are the main failure.** With 3 examples, only 195 of 1,322 listed exercises (15%) are exact catalog names. The other 1,127:
   - 10 differ only in case or spacing (`barbell Romanian deadlift`),
   - 180 are near-misses, string similarity ≥ 0.85 with Python `difflib.SequenceMatcher` (`dumbbell flyes` vs `dumbbell fly`, `dumbbell incline press` vs `dumbbell incline bench press`),
   - 937 are invented (`dumbbell front raise to shoulder height`).
   The 0.85 threshold is strict, so **the near-miss share is a lower bound**: many "invented" names are the right exercise with extra words (`dumbbell lateral raise to shoulder height` vs `dumbbell lateral raise`, ratio 0.70), and 367 of them score between 0.8 and 0.85.
   The model knows the gym vocabulary, not our catalog. Names are the first failed check in 279 of 300 answers.
   To see whether the model still understood the request, we took the closest catalog name for every non-matching name: equipment is right 70–86% of the time, body part only 40–61% (59% even for exact names). **The main body-part mix-up is arms → shoulders**: "arms" requests answered with shoulder exercises 52 times (next: forearms → upper arms, 8). The closest-name check is a rough proxy: a similar string is not always the same exercise.

3. **Sets/reps are not learned, even when an example shows them.** Exactly right: easy 1/324 exercises (0%), medium 32/423 (8%), hard 29/575 (5%). The medium and hard examples in the prompt show 3 × "8-12" and 4 × "5-8", but the model mostly writes the general gym default 3 × "10-12". The prompt has no easy workout example (the easy one is an error answer), so easy is almost always 3 sets. None of the failures is a formatting issue (string vs int, dash type, spaces): the numbers themselves are wrong.

4. **0 of 16 impossible requests get an error answer.** The model always makes up a workout, even though one of the 3 examples is an error answer.

5. **The ~0.41 reward is mostly "free".** It comes from the 10% format bonus, the right count (88%), no duplicates (94%; 3% of answers repeat a name exactly, another 3% list a variant pair like `x` + `x v. 2`) and `advanced_ok` (1.0, helped by the fact that an invented name can never be flagged advanced). `all_correct` is 0 for every answer. The reward measures format copying, not correct workouts: SFT has to teach the catalog names, the sets/reps mapping and the error answer.
