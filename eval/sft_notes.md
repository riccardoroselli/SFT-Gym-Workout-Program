# SFT notes: Qwen3.5-2B after supervised fine-tuning

Model: [`riccardoroselli/qwen3.5-2b-gym-sft`](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-sft) (LoRA r=16 merged, bf16), trained with `notebooks/02_sft_colab.ipynb`.
Same test set as the baselines: 300 requests (unseen combinations, 3 unseen phrasing templates), 16 of them impossible. Greedy decoding, thinking off, zero-shot.
Breakdowns come from `python src/analyze_errors.py --run sft_lora_r16`.

| run | n | parse_ok | strict_format | names_exist | body_part_ok | equipment_ok | count_ok | no_duplicates | sets_reps_ok | advanced_ok | error_correct | all_correct | reward | has_think | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_zeroshot | 300 | 0.0 | 0.0 | - | - | - | - | - | - | - | 0.0 | 0.0 | 0.0 | 0.0 | 0.14 |
| base_fewshot3 | 300 | 0.9967 | 0.9967 | 0.1611 | 0.0915 | 0.1451 | 0.8829 | 0.9732 | 0.0167 | 1.0 | 0.0 | 0.0 | 0.4174 | 0.0 | 0.0 |
| sft_lora_r16 | 300 | 1.0 | 1.0 | 0.7501 | 0.74 | 0.7305 | 1.0 | 0.5068 | 1.0 | 0.9966 | 0.25 | 0.24 | 0.7936 | 0.0 | 0.0 |

First failed check in `sft_lora_r16` (impossible requests count as error handling):
names 171 · duplicates 45 · error handling 12 · passed 72. Nothing fails on parse, body part, equipment, count, sets/reps or advanced.

## Observations

1. **SFT fixed the format, the count and sets/reps.** Every answer is the exact JSON shape, has the requested number of exercises, and uses the right sets/reps for the difficulty (1,312 of 1,312 exercises; the baseline managed 2%). The model also picks the right *kind* of exercise: among exact catalog names, body part is right 99% of the time and equipment 98% (the baseline: 59% body part).

2. **Names are the main failure, and the problem is the exact name, not the type.** 971 of 1,312 listed exercises (74%) are exact catalog names. The other 341 are 244 near-misses (69 unique, e.g. `dumbbell incline chest press` for `dumbbell incline bench press`) and 97 invented (32 unique, e.g. `dumbbell incline fly on exercise ball (with arm blaster)`). Even for those, the closest catalog name has the right body part 95% / 68% of the time. The baseline had 937 invented names; SFT cut that by 90%. Names are the first failed check in 171 answers.

3. **Duplicates, and collapse onto a few favourites.** 146 of 296 workout answers (49%) repeat a name exactly (1.43 extra copies on average); `dumbbell incline fly on exercise ball` alone accounts for 65 extra copies. Counting near-duplicates too (`rear fly` next to `rear fly v. 2`), 176 answers (59%) repeat themselves. Variety is low: across the 33 (body part, equipment) pairs the model uses only 189 of 2,334 valid exercises (8%); for chest / any it uses 2 of 155. It works where the catalog pool is small enough to memorise: **bands score all_correct 0.90, free weights 0.05**. Greedy decoding makes this worse: after one "dumbbell incline fly …", the most likely next item is the same one again.

4. **Refusals: 4 of 16.** All 4 are forearms / bands at easy or medium. Forearms / bodyweight (0 of 5) and shoulders / bodyweight (0 of 6) still get a made-up workout, even though shoulders / bodyweight is the most common impossible combo in train (31 rows). The baseline refused 0 of 16; SFT never refuses a feasible request.

5. **Easy 0.54 vs hard 0.11 is mostly the test mix, then names; not the pool size.**
   - *Pool size:* not the cause. Hard's median pool (60) is smaller than medium's (74), yet hard and medium score about the same (0.11 vs 0.15).
   - *Equipment mix:* the held-out test combos give easy 27 of its 81 rows on bands (vs 5 medium, 9 hard), and bands score about 0.9 at every difficulty. Without bands (feasible rows only): easy 0.35, medium 0.10, hard 0.07.
   - *Number of exercises:* easy asks for 4.05 on average, hard 4.71, and all_correct falls with n (3: 0.51, 4: 0.31, 5: 0.08, 6: 0.03). But at the same n easy is still ahead (n=4: 0.59 vs 0.14).
   - *Impossible rows:* 10 of the 16 are hard, and 10 of hard's failures are failed refusals.
   - *First failed check:* what is left is names: 86% of easy names are exact vs 68% for hard; names are the first failure in 22 easy, 65 medium and 84 hard answers, while duplicates are 15 at every difficulty. Why hard names are worse was not checked.

All three test-only templates score about the same (0.22–0.26), so unseen phrasing doesn't seem to hurt. There is no familiar-template test row to compare against: by design, test uses only the 3 test-only templates.

## What GRPO should target

- **Duplicates and refusals.** These are behaviour, not knowledge: the model knows several valid exercises for most pairs and knows the error format, but doesn't choose them. The verifier already rewards `no_duplicates` and `error_correct` directly, so GRPO can push on them. Near-duplicates (`v. 2`, `(male)`) are distinct names for the verifier today, so they are not penalised: decide whether to count them.
- **Names, partly.** Wrong names are partly a knowledge problem: RL can only push toward names the model already produces sometimes; it can't teach a catalog name the model never samples. GRPO may raise names_exist, but the stronger fix is retrieval (put the matching catalog names in the prompt) or constrained decoding (allow only catalog names). Recorded as future work.
- **all_correct is the headline metric.** The reward gives partial credit (format, count, sets/reps) that the SFT model already earns almost for free. A GRPO run that raises the reward but not all_correct is collecting those free points, not getting better: watch for reward hacking.
