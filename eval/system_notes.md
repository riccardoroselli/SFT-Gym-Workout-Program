# System notes: parser + guard + Q8_0 + verifier retries

The served system is not the model alone. `src/api.py` wraps the Q8_0 GGUF of [`riccardoroselli/qwen3.5-2b-gym-grpo`](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo) (see `gguf_notes.md`) in four steps, all deterministic except the model:

1. **Parser** (`src/request_parser.py`): free text → body part, equipment group, difficulty, count. It uses the spec vocabulary plus serving-only aliases. A request it can't read gets a 422, not a model call.
2. **Guard**: if the catalog has too few matching exercises (`is_feasible()` false), the API returns data-v1's exact error answer without calling the model.
3. **Model**: the greedy answer, exactly as `evaluate.py`'s llamacpp backend produces it.
4. **Verifier retries**: if `verify()` says the answer isn't all correct, the model is asked again, up to `MAX_RETRIES=3` times. Retry *k* uses temperature 0.7 and seed *k*, with llama-server's top-k / top-p / min-p filters off. The first all-correct answer wins; if none passes, the API returns the one with the best reward, the earliest on a tie.

## How it was measured

- `python src/eval_service.py run --run system_q8` sends each of the 300 test requests (the same rows as every other run, 16 of them impossible) to the live API: uvicorn with `MAX_RETRIES=3`, and llama-server `b12360-2d8a32502` with Q8_0 (`-ngl all -c 4096 -np 1`) on the Apple M4 (Metal).
  - The parser read all 300 requests back as their own meta; `run` stops at the first one it doesn't.
  - Each saved response records every model call (`attempts`: text, seed, tokens, ms, reward, first failed check).
- **Greedy is unchanged:** the first call's text is identical to `sft_grpo_gguf_q8` on **284 / 284** model rows. So everything the system adds comes from the guard and the retries, not from a different prompt or decoding.
- `eval_service.py derive` rebuilds **rK**, the answer the service would give with `MAX_RETRIES=K`, for K = 0..3, using the API's own rule on the saved attempts. One run gives all four budgets, and for r3 each derived answer is checked against the live one. Every rK file is scored by the same `evaluate.py score` as the model runs.
- r0 and r3 are in `results/results.md` (marked as system rows); r1 and r2 are in `results/system_q8_r1.json` and `system_q8_r2.json` only.

## Results

| run | names_exist | body_part | equipment | no_duplicates | sets_reps | error_correct | all_correct | reward | model calls per model request |
|---|---|---|---|---|---|---|---|---|---|
| sft_grpo_gguf_q8 (model only) | 0.9549 | 0.9404 | 0.9326 | 0.6599 | 0.9966 | 6/16 | 0.55 | 0.9183 | 1 |
| system_q8_r0 (+ guard) | 0.9575 | 0.9554 | 0.9525 | 0.6655 | 0.9965 | 16/16 | 0.5833 | 0.9483 | 1 |
| system_q8_r1 | 0.9748 | 0.9680 | 0.9687 | 0.7852 | 0.9965 | 16/16 | 0.6667 | 0.9669 | 1.44 |
| system_q8_r2 | 0.9843 | 0.9703 | 0.9803 | 0.8627 | 1.0 | 16/16 | 0.7433 | 0.9780 | 1.79 |
| system_q8_r3 (+ guard + 3 retries) | 0.9904 | 0.9814 | 0.9853 | 0.9049 | 1.0 | 16/16 | **0.82** | **0.9852** | 2.06 |

- **The guard** fixes the 10 impossible requests the model answered with a workout. That is the whole 0.55 → 0.5833 step (+10 / 300). The workout columns move a little from Q8_0 to r0 for the same reason: those 10 answers are no longer workouts, so they no longer count in the workout averages.
- **Each retry adds about 0.08 all_correct** (25, then 23, then 23 more requests), and the gain hasn't levelled off by K=3.
- **No duplicates** rises the most, from 0.66 to 0.90: a sampled answer is a new list, often without the repeat.
- **By difficulty**, all_correct goes from r0 to r3 as follows: easy 0.75 → 0.86, medium 0.58 → 0.88, hard 0.47 → 0.75.

## Rescued vs still failing

125 of the 284 model requests failed at greedy. With up to 3 retries, 71 were rescued and 54 still fail. Their first failed check at greedy:

| first failed (greedy) | rescued | still failing |
|---|---|---|
| duplicates | 37 | 29 |
| names | 32 | 22 |
| equipment | 0 | 3 |
| sets/reps | 1 | 0 |
| advanced | 1 | 0 |
| **total** | **71** | **54** |

A greedy duplicate is not "fixed" by sampling the same habit again: in 15 of the 29 still-failing duplicate rows, all three retries failed on **names** instead. Temperature 0.7 removes the repeat, but it also brings back names that aren't in the catalog.

## Latency (live answers, `MAX_RETRIES=3`)

Total time per request inside the API, by the attempt that answered and whether it was verified. An unverified row made all 4 calls; its source is the attempt with the best reward.

| source | verified | rows | mean ms | p50 | p95 |
|---|---|---|---|---|---|
| guard | true | 16 | 0 | 0 | 1 |
| model_greedy | true | 159 | 2,737 | 2,580 | 3,995 |
| model_greedy | false | 32 | 13,654 | 13,089 | 17,923 |
| model_retry_1 | true | 25 | 7,409 | 7,447 | 9,332 |
| model_retry_1 | false | 10 | 15,294 | 15,069 | 18,112 |
| model_retry_2 | true | 23 | 10,336 | 10,211 | 12,898 |
| model_retry_2 | false | 8 | 12,608 | 11,945 | 16,012 |
| model_retry_3 | true | 23 | 13,293 | 12,491 | 17,322 |
| model_retry_3 | false | 4 | 13,793 | 13,646 | 18,324 |

Averaged over all 284 model requests, as derived per budget:

| budget | r0 | r1 | r2 | r3 |
|---|---|---|---|---|
| mean ms | 3,085 | 4,604 | 5,798 | 6,725 |
| max ms | 5,256 | 10,072 | 14,932 | 19,644 |

A greedy pass takes about 2.7 s. Every failed attempt adds another call of about 3 s. The 54 requests that never pass cost the most, about 13–15 s, and the user still gets an unverified answer (marked `"verified": false`).

## What retries can't fix

- **Names the model never learned.** In 11 of the 22 still-failing *names* rows, every one of the 4 attempts has at least one name that isn't in the catalog. The same wrong names come back across attempts and requests:
  - `bodyweight standing archer` (6 times; closest catalog name `bodyweight standing row`);
  - `dumbbell lying one arm supination` (4);
  - `barbell decline close grip bench press` (4);
  - `dumbbell one arm hammer curl on exercise ball` (3).

  They are near-misses: catalog names mixed together, which the model treats as real. Arms with free weights is the worst combination, 18 of the 54 still-failing rows.
- **3 equipment rows, all *hard abs with free weights, 5 exercises*.** All 4 attempts include `band …` exercises (`band jack knife sit-up`, `band v-up`, …), even though 30 catalog exercises match. This isn't a small pool: the model ties abs to bands. Sampling at temperature 0.7 doesn't break that habit.
- Both are knowledge gaps in the weights. More retries make a lucky answer more likely, but only more training (or constrained decoding to catalog names) removes them.

## Caveats

- **This is a system, not a model.** r0–r3 are not comparable with the model rows in `results.md` as a measure of the model. They measure what a user of the API gets. The model's own score is still `sft_grpo_gguf_q8`: 0.55.
- **`MAX_RETRIES=3` was fixed in advance** (in the Step 16a design) and not tuned on the test set. r1 and r2 are reported to show the curve, not to pick a budget after looking at test results.
- **Retries cost latency.**
  - A request makes 2.06 model calls on average instead of 1.
  - The mean time per model request goes from 3.1 s to 6.7 s, and the slowest requests take up to about 20 s.
  - This was measured on Metal on the M4. On a CPU (Docker) each call is several times slower, and the retry cost grows with it.
- **The seeds only repeat on the same setup.** The same seed gives the same text on the same llama.cpp build, model file and hardware (checked: two identical calls gave identical text). A different build, backend (CPU vs Metal) or quantization can change the sampled retries, so r1–r3 may move by a few requests. Greedy (r0) is less sensitive, but it isn't guaranteed bit-identical either (see `gguf_notes.md`: MLX vs llama.cpp).
- **One test set of 300 requests**, with no confidence intervals here. The r0 → r3 gain (+71 requests) is far larger than the noise seen between engines in `gguf_notes.md`.

## Decision

Serve with the guard and `MAX_RETRIES=3`: **0.82 all_correct, 0.985 reward**, and every answer says whether it passed the verifier. The cost is about 2 model calls per request.
