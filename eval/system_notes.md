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
  - This was measured on Metal on the M4. On the CPU in Docker a model request took about 1.7–1.8× as long (see Docker (CPU) below), and the retry cost grows with it.
- **The seeds only repeat on the same setup.** The same seed gives the same text on the same llama.cpp build, model file and hardware (checked: two identical calls gave identical text). A different build, backend (CPU vs Metal) or quantization can change the sampled retries, so r1–r3 may move by a few requests. Greedy (r0) is less sensitive, but it isn't guaranteed bit-identical either (see `gguf_notes.md`: MLX vs llama.cpp).
- **One test set of 300 requests**, with no confidence intervals here. The r0 → r3 gain (+71 requests) is far larger than the noise seen between engines in `gguf_notes.md`.

## Decision

Serve with the guard and `MAX_RETRIES=3`: **0.82 all_correct, 0.985 reward**, and every answer says whether it passed the verifier. The cost is about 2 model calls per request.

## Docker (CPU)

Step 16c packs the same system into two containers (now `docker-compose.dev.yml`). The API code and the model file are the same as above, but the llama.cpp backend is different: CPU instead of Metal.

### Setup
- **`llama`:** llama-server built from source at the tag `b12360` (`https://github.com/ggml-org/llama.cpp.git#b12360`, `.devops/cpu.Dockerfile`, target `server`).
  - It's a portable CPU build (`GGML_NATIVE=OFF`, `GGML_CPU_ALL_VARIANTS=ON`): every arm64 CPU variant is compiled, and the best one is picked at start.
  - It runs with `-c 4096 -np 1` and no `-ngl` (CPU only), with 10 threads.
  - The Q8_0 GGUF is mounted read-only from `models/gguf`, and the service has no host port.
- **`api`:**
  - `python:3.12-slim` with the 7 pinned packages of `requirements-serve.txt`.
  - The tokenizer files come from `Qwen/Qwen3.5-2B` at revision `15852e8`. They give the same prompt text and token IDs as the fine-tuned repo on 2,980 prompts (2,500 data-v1 + 480 canonical sentences).
  - `HF_HUB_OFFLINE=1` and `MAX_RETRIES=3`. Only `127.0.0.1:8000` is published.
- **Docker Desktop** on the M4 MacBook Air: linux/arm64, 10 CPUs, 7.75 GiB of memory.

### Version check
`docker compose exec llama /app/llama-server --version` prints `version: 1.0.0-dev (build 1, commit 2d8a325)`, built with GNU 15.2.0 for Linux aarch64.
- **The commit is the tag `b12360` (`2d8a32502909…`), the same code as the Mac build.**
- The build number is 1, not 12360. llama.cpp counts the commits in its git history, and Docker's git context fetches only the tagged commit. So `/health` and every response report `server_build: "b1-2d8a325"`, where the Mac reports `"b12360-2d8a32502"`.

### Live requests (18 Nov 2026, through `127.0.0.1:8000`)

| request | parsed meta | HTTP | source | verified | model calls | reward | total ms | tok/s per call | interpreted_as |
|---|---|---|---|---|---|---|---|---|---|
| forearms workout, easy, bands, 3 exercises | forearms / bands / easy / 3, impossible | 200 | guard | true | 0 | 1.0 | 1 | – | – |
| chest workout, medium difficulty, free weights, 5 exercises | chest / free weights / medium / 5 | 200 | model_greedy | true | 1 | 1.0 | 3,556 | 33.3 | – |
| shoulders, bodyweight, easy | shoulders / bodyweight / easy / 4, impossible | 200 | guard | true | 0 | 1.0 | 0 | – | – |
| two exercises for arms | – (missing equipment, difficulty; count 2) | 422 | – | – | – | – | – | – | – |
| biceps workout, easy, dumbbells, 5 exercises | arms / free weights / easy / 5 | 200 | model_retry_1 | true | 2 | 1.0 | 7,628 | 33.7, 33.6 | biceps → arms; dumbbells → free weights |
| hard legs workout with machines, 6 exercises | legs / machines / hard / 6 | 200 | model_greedy | true | 1 | 1.0 | 3,618 | 34.0 | – |
| abs, any equipment, intense, three exercises | abs / any / hard / 3 | 200 | model_greedy | true | 1 | 1.0 | 1,998 | 34.0 | – |
| calves workout, easy, bands, 5 exercises | calves / bands / easy / 5, impossible | 200 | guard | true | 0 | 1.0 | 0 | – | – |

- `/health`: `{"status":"ok","model_file":"qwen3.5-2b-gym-grpo-Q8_0.gguf","server_build":"b1-2d8a325"}`.
- The answers have the same sources as on Metal (Step 16a). The biceps greedy answer failed on duplicates again, and retry 1 passed. Each model request took about 1.3× as long as on Metal.
- **`docker stats` during the chest request:**
  - `llama`: 926 % CPU (about 9.3 of the 10 CPUs) and 4.11 GiB of the 7.75 GiB, which includes the 2 GB model mapped into memory;
  - `api`: 4 % CPU and 317 MiB.

### 30 test rows: Docker vs Metal
I ran `eval_service.py run --run docker_q8 --limit 30`, then `derive` and `report`. The Metal side is the first 30 rows of the Step 16b run (`system_q8`), with the same API code and the same model file. 1 of the 30 rows is impossible (the guard answers it), so 29 are model rows.

**Greedy text identical to `sft_grpo_gguf_q8`: 22 / 29.** The 7 rows that differ (test row numbers, 0-based):

| row | request | first difference | Docker | Metal | Docker reward (first failed) | Metal reward (first failed) |
|---|---|---|---|---|---|---|
| 2 | light pecs, weights, 5 | exercise 1, sets | 3 | 2 | 0.82 (duplicates) | 0.91 (duplicates) |
| 5 | medium calves, machines and cables, 6 | exercise 4, name | lever calf press | smith toe raise | 0.91 (duplicates) | 0.91 (duplicates) |
| 9 | medium abs, free weights, 5 | exercise 1, name | barbell seated twist | barbell sitted alternate leg raise | 0.91 (duplicates) | 0.802 (names) |
| 13 | light back, bands, 4 | exercise 4, name | band one arm twisting row | band straight leg deadlift | 0.865 (names) | **1.0 (passed)** |
| 16 | lower arms, machines, intermediate, 6 | exercise 1, name | cable reverse wrist curl | cable wrist curl | 0.91 (duplicates) | 0.91 (duplicates) |
| 19 | light chest, any equipment, 5 | exercise 2, name | cable one arm fly on exercise ball | cable decline fly | 1.0 (passed) | 1.0 (passed) |
| 20 | hard shoulders, any equipment, 6 | exercise 5, name | dumbbell rear fly v. 2 | dumbbell one arm lateral raise | 0.82 (names) | 0.82 (names) |

| first 30 rows | Docker (CPU) | Metal (Step 16b) |
|---|---|---|
| r0 all_correct | 17 / 30 (0.5667) | 18 / 30 (0.60) |
| r0 reward | 0.9454 | 0.9493 |
| r3 all_correct | 25 / 30 (0.8333) | 25 / 30 (0.8333) |
| r3 reward | 0.9858 | 0.9858 |
| model calls per model request (r3) | 2.0 | 2.03 |
| mean ms per model request, r0 / r3 | 4,002 / 8,503 | 2,266 / 5,007 |
| greedy tok/s, mean (min–max) | 26.3 (14.7–34.5) | 44.1 (43.0–44.8) |

Latency of the live answers (`MAX_RETRIES=3`), same 30 rows:

| source | verified | Docker rows | Docker mean / p50 / p95 ms | Metal rows | Metal mean / p50 / p95 ms |
|---|---|---|---|---|---|
| guard | true | 1 | 0 / 0 / 0 | 1 | 0 / 0 / 0 |
| model_greedy | true | 16 | 3,674 / 3,415 / 7,044 | 17 | 2,004 / 1,823 / 2,968 |
| model_greedy | false | 3 | 19,481 / 23,080 / 24,297 | 4 | 11,065 / 11,483 / 12,247 |
| model_retry_1 | true | 5 | 9,197 / 9,046 / 13,588 | 2 | 6,084 / 5,750 / 6,418 |
| model_retry_2 | true | – | – | 2 | 8,129 / 7,440 / 8,818 |
| model_retry_2 | false | 1 | 15,278 | – | – |
| model_retry_3 | true | 3 | 18,358 / 16,634 / 22,986 | 3 | 10,162 / 10,276 / 11,912 |
| model_retry_3 | false | 1 | 13,014 | 1 | 7,981 |

### What differs from Metal, and why
- **Greedy text on 7 of 29 rows.** The CPU and Metal kernels add up the same numbers in a different order, so the logits differ in the last digits. Where two tokens are almost tied, greedy picks the other one, and everything after it changes. In 3 rows this already happens in the first exercise.
  - Neither backend is better. Row 13 passes on Metal only, and row 9 has a higher reward on Docker. Net: r0 is 17 vs 18 out of 30.
- **Retries diverge more.** The same seed gives the same sample only if the logits are the same. Only 8 of 27 retry texts that both runs produced are identical.
  - r3 is still 25 / 30 on both, but not on the same rows: one row passes only on Docker and one only on Metal.
- **Speed.** Fresh, the CPU makes about 34 tok/s against Metal's 44.
  - During the 30 rows it fell steadily to 15–20 tok/s, while Metal stayed at 43–45 on the same rows.
  - The likely cause is thermal throttling: all 10 CPU cores ran for minutes on a fanless MacBook Air. I didn't measure temperature. Metal slows down too over a long run (mean 34.3 tok/s over all 300 Step 16b rows), but only after the first 30 rows.
  - Mean time per model request: 1.8× Metal at r0 and 1.7× at r3. The slowest Docker request took 24.3 s.
- **What this means:** the Step 16b score (0.82 all_correct) is a Metal number. On these 30 rows Docker reaches the same level but not the same answers row by row. 30 rows are too few to give Docker its own score, and the run wasn't scored into `results.md`.

### The single image (Step 17.2)
The all-in-one image (`docker/Dockerfile`: llama-server b12360 built for the CPU, the API and the page; the split Q8_0 is downloaded on the first start) on the same Mac:
- Image 623 MB on disk. `llama-server --version`: build 12360, commit 2d8a32502. `docker stats` during the live requests: 4.44 GiB.
- Rule fixed before the run: the image builds llama-server with GCC 14 (Debian), not with llama.cpp's own `.devops/cpu.Dockerfile` as in 16c, so the 30-row run passes only with no errors, differing greedy rows that look like near-tie flips, and r3 all_correct within ±3 of 25 / 30 (16c). It passed.
- `eval_service.py run --run docker_aio_q8 --limit 30`: r0 19 / 30, r3 26 / 30 all correct (7 rescued), greedy p50 3.0 s. The greedy text equals Metal's on 19 / 29 model rows (16c: 22 / 29); where it differs, it starts mid-answer, a near-tie flip as above.
- First start: the 2.1 GB download took about 3 min (~11 MB/s); from model load to ready took 2.5 s.

### How to run
Three ways, same API and page at http://127.0.0.1:8000 (`/docs` for the API).

**1. The single image (llama-server + API + page; Step 17.2).** The first start downloads the model (2.1 GB) into the `gymleader-models` volume; follow it with `docker logs -f gymleader` or `docker compose logs -f`.
```bash
docker run -d --name gymleader -p 127.0.0.1:8000:8000 -v gymleader-models:/models ghcr.io/riccardoroselli/gymleader:v1.0
# or, from this checkout (pulls the image, or builds docker/Dockerfile if it isn't published):
docker compose up -d
docker compose ps                                             # (healthy)
docker exec gymleader llama-server --version                  # build 12360, commit 2d8a32502 (compose: docker compose exec gymleader ...)
docker compose down                                           # the volume and the model stay
```

**2. The two-container dev setup of this step (llama.cpp's own image + `docker/api.Dockerfile`, `./models/gguf` mounted).**
```bash
docker compose -f docker-compose.dev.yml build               # first time: compiles llama.cpp (several minutes)
docker compose -f docker-compose.dev.yml up -d
docker compose -f docker-compose.dev.yml ps                  # both (healthy)
docker compose -f docker-compose.dev.yml exec llama /app/llama-server --version  # build 1, commit 2d8a325
docker compose -f docker-compose.dev.yml down
```

**3. On the Mac without Docker (Metal),** in the env with `requirements-serve.txt`: `scripts/run_local.sh` (Ctrl-C stops both processes). It uses `models/gguf/` and the `llama-server` on PATH, and warns if it isn't build 12360.

**Requests and the 30-row check** (any of the three):
```bash
curl -s 127.0.0.1:8000/health
curl -s 127.0.0.1:8000/workout -H 'Content-Type: application/json' \
     -d '{"request": "chest workout, medium difficulty, free weights, 5 exercises"}'
python src/eval_service.py run --run docker_q8 --limit 30   # then derive / report --run docker_q8
```
