# GGUF notes: the GRPO model, converted and quantized for llama.cpp

Model: [`riccardoroselli/qwen3.5-2b-gym-grpo`](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo) (private; see `grpo_notes.md`), converted to GGUF with llama.cpp b12360 and quantized to Q8_0 and Q4_K_M.
Same test set as every other run: 300 requests (unseen combinations, 3 unseen phrasing templates), 16 of them impossible. Greedy decoding, thinking off, zero-shot.
Answers come from `python src/evaluate.py generate --backend llamacpp`, breakdowns from `python src/analyze_errors.py --run <run>`, paired comparisons from `python src/compare_runs.py <A> <B>`.

## Why GGUF

Step 16 serves the model from a Docker container. Until now every answer came from MLX, which runs on the Apple GPU through Metal; a Linux container has no Metal, so it can't use MLX. llama.cpp runs GGUF files on a plain CPU (and on Metal, CUDA and other GPUs), so the same file works on the Mac and in Docker. GGUF also stores smaller, quantized weights: fewer bits per weight means a smaller file and faster writing, since generating each token reads every weight once.

## Conversion and quantization

`convert_hf_to_gguf.py` (text model only, no vision projector) → BF16 GGUF → `llama-quantize` → Q8_0 and Q4_K_M. Architecture `qwen35`, 299 tensors, 25 blocks (24 layers + 1 multi-token-prediction block), 1.94 B parameters, end-of-turn token `<|im_end|>`. The tokenizer gives exactly the same token IDs as the Hugging Face one on the evaluation prompt.

| file | size | bits per weight | tensor types |
|---|---|---|---|
| safetensors (Hub) | 4.55 GB | 16 | bf16 (includes the vision tower) |
| BF16 | 3.90 GB | 16.0 | 159 BF16, 140 F32 |
| Q8_0 | 2.08 GB | 8.5 | 159 Q8_0, 140 F32 |
| Q4_K_M | 1.32 GB | 5.4 | 132 Q4_K, 27 Q6_K, 140 F32 |

The 140 F32 tensors are small (norms, biases, 0.2% of the bytes) and are never quantized.

**Why Q4_K_M is 5.4 bits per weight, not 4.** Q8_0 stores 8-bit numbers plus one scale per block of 32, so 8.5 bits. Q4_K stores 4-bit numbers plus scales and minimums per small block, so 4.5 bits. The "_M" (medium) mix keeps the tensors that hurt most when rounded at Q6_K, 6.56 bits: the token embeddings, 13 of the 25 `ffn_down` matrices, 10 `attn_qkv_gate` and 3 `attn_v` matrices. The token embeddings alone are 509 M parameters (248k-token vocabulary × 2048), 26% of the model, because Qwen3.5 has a very large vocabulary and a small body. All Q6_K tensors together are 43% of the parameters, so the average is 0.57 × 4.5 + 0.43 × 6.56 ≈ 5.4.

## How it was evaluated

- One `llama-server` at a time (`build_info` `b12360-2d8a32502`, `-ngl all -c 4096 -np 1`), started with the full path of the file. `evaluate.py` checks that the server loaded the `--model` file and has at least 2048 tokens of context.
- The prompt is exactly `build_prompt()` with the Hugging Face tokenizer, sent raw to `/completion` (no chat template on the server). Settings: greedy, 512 new tokens at most, stop at `<|im_end|>`, no prompt cache, every repetition penalty explicitly off, so the answers come from the same decoding as MLX's.
- Every row records the backend, the GGUF file name and the server build. 300 rows per file, 0 truncated, 0 empty. Scored by the same `evaluate.py score` and verifier.
- **Reference for quantization: the GGUF BF16 run, not the MLX one.** Comparing Q8_0 and Q4_K_M with GGUF BF16 measures the quantization alone, without the change of engine mixed in.

| run | parse_ok | names_exist | body_part | equipment | no_duplicates | sets_reps | error_correct | all_correct | reward |
|---|---|---|---|---|---|---|---|---|---|
| sft_lora_r16 (MLX) | 1.0 | 0.7501 | 0.74 | 0.7305 | 0.4054 | 1.0 | 4/16 | 0.24 | 0.7846 |
| sft_grpo (MLX) | 1.0 | 0.9477 | 0.9338 | 0.9255 | 0.6633 | 0.9966 | 6/16 | 0.55 | 0.9146 |
| sft_grpo_gguf_bf16 | 1.0 | 0.9535 | 0.9391 | 0.9312 | 0.6497 | 0.9966 | 6/16 | 0.54 | 0.9166 |
| sft_grpo_gguf_q8 | 1.0 | 0.9549 | 0.9404 | 0.9326 | 0.6599 | 0.9966 | 6/16 | 0.55 | 0.9183 |
| sft_grpo_gguf_q4km | 1.0 | 0.8152 | 0.8030 | 0.7991 | 0.5430 | 0.9897 | 7/16 | 0.2467 | 0.8340 |

Full rows in `results/results.md`.

| `analyze_errors.py` | MLX | BF16 | Q8_0 | Q4_K_M |
|---|---|---|---|---|
| first failed: passed | 165 | 162 | 165 | 74 |
| first failed: names | 60 | 57 | 54 | 161 |
| first failed: duplicates | 60 | 66 | 66 | 52 |
| first failed: equipment / body part | 3 / 0 | 3 / 0 | 3 / 0 | 0 / 1 |
| first failed: sets/reps / advanced | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 2 |
| first failed: error handling | 10 | 10 | 10 | 9 |
| names exact / near-miss / invented | 1224 / 44 / 35 (of 1303) | 1232 / 43 / 28 (of 1303) | 1233 / 40 / 30 (of 1303) | 1042 / 157 / 94 (of 1293) |
| duplicates (exact or variant) | 99/294 (34%) | 103/294 (35%) | 100/294 (34%) | 133/291 (46%) |
| impossible refused | 6/16 | 6/16 | 6/16 | 7/16 |
| false refusals (of 284 feasible) | 0 | 0 | 0 | 2 |
| all_correct easy / medium / hard | 0.75 / 0.55 / 0.41 | 0.73 / 0.61 / 0.36 | 0.75 / 0.56 / 0.41 | 0.47 / 0.22 / 0.12 |

**Paired comparison.** All runs answered the same 300 requests, so each request is compared with itself. The 95% intervals come from a paired bootstrap (resample the 300 requests 10,000 times, seed 42); if an interval contains 0, the difference is within noise.

| A → B | identical answers | differing: B reward higher / lower / equal | all_correct gained / lost | Δ reward [95% CI] | Δ all_correct [95% CI] |
|---|---|---|---|---|---|
| MLX → BF16 (engine) | 201/300 | 19 / 18 / 62 | 11 / 14 | +0.0021 [−0.0023, +0.0066] | −0.010 [−0.043, +0.023] |
| BF16 → Q8_0 | 228/300 | 16 / 12 / 44 | 11 / 8 | +0.0016 [−0.0029, +0.0062] | +0.010 [−0.020, +0.040] |
| BF16 → Q4_K_M | 12/300 | 17 / 157 / 114 | 4 / 92 | −0.0827 [−0.0993, −0.0664] | −0.293 [−0.350, −0.240] |

## Observations

1. **Changing engine is noise.** The same bf16 weights in llama.cpp give the same answer as in MLX for 201 of 300 requests. The other 99 split evenly (19 better, 18 worse, 62 equal reward) and both intervals contain 0. The differences start partway through an answer, at the choice of the next exercise. Likely cause (not measured): the two engines round their arithmetic slightly differently, which is enough to flip greedy decoding where two exercises are almost equally likely. A 5-request smoke test had matched 5 of 5; that was luck.

2. **Q8_0 is BF16 within noise.** 228 of 300 answers identical, 16 better and 12 worse, both intervals contain 0, and every row of the error table is within a few answers of BF16. At half the size it writes 1.7× faster.

3. **Q4_K_M loses most of what GRPO added.** all_correct 0.54 → 0.25, back to about the SFT level (0.24); the reward, 0.834, stays above SFT's 0.785. It loses 92 requests that BF16 got right and gains 4. Only 12 of 300 answers are the same as BF16's.
   - **Names** are the main failure (first failed check in 161 answers, up from 57): 81% exact catalog names, against 95%. It paraphrases again (`dumbbell incline press` for `dumbbell incline bench press`, 157 near-misses) and invents more (94).
   - **Duplicates** rise from 35% to 46% of workouts.
   - **2 false refusals** of feasible requests, which no GRPO run had made; 1 more impossible request refused (7 of 16).
   - **Hard requests** fall to 0.12 (BF16 0.36).
   - **Hypothesis (not measured):** GRPO changed the weights by small, precise amounts. On a 2B model, 4-bit rounding is coarse enough to blur them, so the model falls back toward its SFT behaviour: the paraphrases and repeats that GRPO had trained away come back.

## Speed

`llama-bench` on the Apple M4 (Metal), all layers on the GPU, 3 repetitions, 60 s pause between files:

| file | size | read prompt, pp512 (tok/s) | write, tg128 (tok/s) |
|---|---|---|---|
| BF16 | 3.90 GB | 957.6 ± 0.7 | 25.5 ± 0.4 |
| Q8_0 | 2.08 GB | 918.0 ± 1.2 | 43.2 ± 0.5 |
| Q4_K_M | 1.32 GB | 892.3 ± 0.6 | 62.9 ± 0.2 |

Reading a prompt is limited by computation and costs about the same for all three. Writing is limited by how fast the weights are read from memory, so it speeds up as the file shrinks. MLX bf16 wrote about 25 tok/s on the same Mac.
During the full test runs Q4_K_M averaged only 45.6 tok/s. It ran last, right after the BF16 and Q8_0 runs; the MacBook Air has no fan, so it slows down when hot (thermal throttling). BF16 (24.6) and Q8_0 (38.5) ran earlier and lost less.

## Decision

- **Serve Q8_0 in Step 16:** the same quality as BF16, half the size, 1.7× faster writing.
- **Publish BF16 and Q8_0.** BF16 is the reference; Q8_0 is the file to use. Q4_K_M is reported here but not published: it's the fastest, but it gives up most of the GRPO gain.

## Limits

- **One test set** of 300 requests: enough to separate Q4_K_M from the others, not to rank BF16 and Q8_0, whose differences are within noise.
- **Greedy decoding only.** With sampling the engine and quantization effects might look different.
- **Speeds were measured on Metal on this Mac**, not in Docker on a CPU, where Step 16 will run.
- The two explanations above (rounding differences between engines, 4-bit rounding blurring GRPO's changes) are **hypotheses, not measured**.

## What's next

- **Step 16:** serve Q8_0 from Docker with llama.cpp.
- **A smaller file that keeps the GRPO gain:** an importance-matrix (imatrix) quantization, which uses sample text to round the most important weights more carefully, or Q5_K_M / Q6_K. Each would need the same full evaluation against GGUF BF16.
