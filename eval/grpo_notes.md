# GRPO notes: the SFT model after reinforcement learning

Model: [`riccardoroselli/qwen3.5-2b-gym-grpo`](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo) (private; the SFT LoRA trained on with GRPO for 300 steps, then merged, bf16), trained with `notebooks/03_grpo_colab.ipynb`. Reward = `verify()["reward"]`, 8 answers per prompt at temperature 0.7, 600 train prompts (20% impossible).
Same test set as the baselines and SFT: 300 requests (unseen combinations, 3 unseen phrasing templates), 16 of them impossible. Greedy decoding, thinking off, zero-shot.
Breakdowns come from `python src/analyze_errors.py --run sft_grpo`, compared with `--run sft_lora_r16`.

| run | n | parse_ok | strict_format | names_exist | body_part_ok | equipment_ok | count_ok | no_duplicates | sets_reps_ok | advanced_ok | error_correct | all_correct | reward | has_think | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_zeroshot | 300 | 0.0 | 0.0 | - | - | - | - | - | - | - | 0.0 | 0.0 | 0.0 | 0.0 | 0.14 |
| base_fewshot3 | 300 | 0.9967 | 0.9967 | 0.1611 | 0.0915 | 0.1451 | 0.8829 | 0.9431 | 0.0167 | 1.0 | 0.0 | 0.0 | 0.4147 | 0.0 | 0.0 |
| sft_lora_r16 | 300 | 1.0 | 1.0 | 0.7501 | 0.74 | 0.7305 | 1.0 | 0.4054 | 1.0 | 0.9966 | 0.25 | 0.24 | 0.7846 | 0.0 | 0.0 |
| sft_grpo | 300 | 1.0 | 1.0 | 0.9477 | 0.9338 | 0.9255 | 1.0 | 0.6633 | 0.9966 | 0.9966 | 0.375 | 0.55 | 0.9146 | 0.0 | 0.0 |

First failed check (impossible requests count as error handling), SFT → GRPO:
names 171 → 60 · duplicates 45 → 60 · error handling 12 → 10 · equipment 0 → 3 · sets/reps 0 → 1 · advanced 0 → 1 · **passed 72 → 165**.

## Observations

1. **all_correct more than doubled: 0.24 → 0.55, not just the reward.** Reward rose 0.785 → 0.915, and all_correct rose with it, so GRPO did not just collect the partial credit SFT already earned (the reward-hacking risk named in `sft_notes.md`). The gain is everywhere except bands, which were already at 0.90:

   | split | SFT | GRPO |
   |---|---|---|
   | easy / medium / hard | 0.54 / 0.15 / 0.11 | 0.75 / 0.55 / 0.41 |
   | free weights / bodyweight / any / machines / bands | 0.05 / 0.09 / 0.17 / 0.25 / 0.90 | 0.34 / 0.38 / 0.67 / 0.57 / 0.90 |
   | n = 3 / 4 / 5 / 6 | 0.51 / 0.31 / 0.08 / 0.03 | 0.89 / 0.60 / 0.45 / 0.22 |
   | the 3 test templates | 0.22–0.26 | 0.49–0.58 |

   Longer workouts are still much harder (n=6: 0.22), because each extra exercise is another chance of a wrong name or a repeat.

2. **Names: fixed by sticking to known names, not by learning new ones.** 1,224 of 1,303 listed exercises (94%) are exact catalog names, up from 74%. Near-misses fell 244 → 44 (69 → 19 unique) and invented names 97 → 35 (32 → 17 unique). But the model now uses *fewer* distinct names (223 vs 263), and 92% of the names it lists appear in train answers (SFT: 74%). GRPO taught the model to stop paraphrasing (`dumbbell incline chest press` for `dumbbell incline bench press`) and to repeat names it learned in SFT. It didn't teach it new names; see "The names-as-knowledge limit" below. The invented names that remain are further off: their closest catalog name has the right body part only 49% of the time (SFT: 68%).

3. **Duplicates: much better, still the joint top failure.** Workout answers with an exact repeat fell from 146 / 296 to 91 / 294 (49% → 31%), with a variant pair 54 → 33 (18% → 11%), and with either 176 → 99 (59% → 34%); `no_duplicates` 0.41 → 0.66. SFT's favourite repeat, `dumbbell incline fly on exercise ball`, fell from 65 extra copies to 9; the new top one is `dumbbell rear fly` (20). Duplicates are now the first failed check in 60 answers, as many as names. They count more there partly because names fail less often now and no longer get counted first. In training (sampling at 0.7) only 10–20% of answers repeated a name, so greedy decoding still falls into the "same exercise again" loop more often than sampling does. Variety barely moved: 221 of 2,334 valid (pair, exercise) slots used (9%, SFT 8%); arms / any uses 4 of 287.

4. **Refusals: 4 → 6 of 16, no false refusals.**

   | impossible combo (test) | rows | SFT refused | GRPO refused |
   |---|---|---|---|
   | forearms / bands / easy, medium | 4 | 4 | 4 |
   | forearms / bands / hard | 1 | 0 | 1 |
   | forearms / bodyweight / hard | 3 | 0 | 1 |
   | forearms / bodyweight / medium | 2 | 0 | 0 |
   | shoulders / bodyweight / hard | 6 | 0 | 0 |

   Neither model refuses a feasible request (0 of 284). Shoulders / bodyweight is still never refused (0 of 6), as the pre-training probe predicted: at temperature 0.7 the SFT model refused it only 1 time in 8, so GRPO had almost no refusal to reward. The test has no chest / bands or calves / bands rows.

5. **Small new failures (5 answers).** 3 abs / free weights / hard answers list band exercises (`band v-up`, `band jack knife sit-up`), one legs / machines / easy answer uses 3 sets instead of 2 (easy sets/reps 313 / 316 exercises), one abs / bodyweight / medium answer includes an advanced exercise. Body-part mix-ups among exact names are as before (shoulders → chest 8, SFT 6).

## The refusal dial (from training, `docs/steps.md` Step 14.3)

The 40-row val check during training shows refusals moving as **one dial for all requests**, not as knowledge of which pairs are impossible:

| step | impossible refused (13) | feasible falsely refused (27) |
|---|---|---|
| 0 (SFT) | 3 | 0 |
| 150 | 6 | 1 |
| 200 | 10 | 3 |
| 250 | 12 | 3 |
| 300 (final) | 3 | 0 |

When refusals went up, false refusals came with them. In training steps 251–275, false refusals reached 20% of feasible answers. A false refusal scores 0.1 against about 0.8 for a workout, and impossible prompts are in only about 1 step in 3, so the push against refusing won and turned refusals down everywhere. By step 300 they were back at SFT's level. The test result (6 / 16) fits this: the final model is at the "low" setting of the dial. We evaluate the final model, not step 250, because the test is 5% impossible: there step 300's better workouts outweigh step 250's refusals.
Likely causes (not measured): a constant learning rate, and too few impossible prompts per step for a stable signal.

## The names-as-knowledge limit

A wrong name is partly missing knowledge, and RL can't add knowledge: it can only make the model choose, more often, answers it already produces sometimes. GRPO did exactly that. It pushed the model toward names from its SFT training data (92% of the names it lists) and away from paraphrases. That is why names_exist rose to 0.95 while variety stayed at 9% of the valid exercises. For a request the train answers never covered well, the model has no better name to choose. The 79 non-catalog names that remain, and the narrow choice of exercises, need a different fix: retrieval (put the matching catalog names in the prompt) or constrained decoding (allow only catalog names). This stays future work.

## What's next

- **Refusals:** the reward can't make the dial stable with these settings. Options: more impossible prompts per step, a decaying learning rate, or, since feasibility can be computed exactly from the catalog (`is_feasible()`), a rule-based check before or after the model.
- **Duplicates:** 34% of workouts still repeat themselves with greedy decoding. Options: more GRPO steps, sampling at inference, or a deduplicating post-check.
- **Names and variety:** retrieval or constrained decoding (see above).
