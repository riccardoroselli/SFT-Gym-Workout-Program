# GGUF notes: the GRPO model, converted and quantized for llama.cpp

Model: [`riccardoroselli/qwen3.5-2b-gym-grpo`](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo) (public; see `grpo_notes.md`), converted to GGUF with llama.cpp b12360 and quantized to Q8_0 and Q4_K_M.
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
| first failed: names | 60 | 57 | 54 | 161¹ |
| first failed: duplicates | 60 | 66 | 66 | 52 |
| first failed: equipment / body part | 3 / 0 | 3 / 0 | 3 / 0 | 0 / 1 |
| first failed: sets/reps / advanced | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 2 |
| first failed: error handling | 10 | 10 | 10 | 9 |
| names exact / near-miss / invented | 1224 / 44 / 35 (of 1303) | 1232 / 43 / 28 (of 1303) | 1233 / 40 / 30 (of 1303) | 1042 / 157 / 94 (of 1293) |
| duplicates (exact or variant) | 99/294 (34%) | 103/294 (35%) | 100/294 (34%) | 133/291 (46%) |
| impossible refused | 6/16 | 6/16 | 6/16 | 7/16 |
| false refusals (of 284 feasible) | 0 | 0 | 0 | 2 |
| all_correct easy / medium / hard | 0.75 / 0.55 / 0.41 | 0.73 / 0.61 / 0.36 | 0.75 / 0.56 / 0.41 | 0.47 / 0.22 / 0.12 |

¹ Counted before `analyze_errors.py` had a "refused" label: 2 of the 161 are the 2 false refusals in the row below. With today's code the Q4_K_M column reads names 159, refused 2. No other column has a false refusal.

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

## Chat-ready files and compatibility

Everything above was measured on the files exactly as converted, with `build_prompt()` sent raw to `/completion`. The published files were then changed twice, so that a chat UI or a chat API gets the same answers (Step 17.1). The tensor bytes and the tokenizer never changed.

### Why
- **Thinking was on by default.** llama-server b12360 runs the GGUF's Jinja template with `--reasoning auto`, which sends `enable_thinking=true` whenever the template supports it. Qwen3.5's stock template then ends the prompt with an open `<think>\n`, which the model never saw in training.
- **No system prompt.** The stock template adds none, while every training example had our `SYSTEM` message.
- **Fused tensors after b12343.** Since llama.cpp b12343 (PR #32102, 12 Nov 2026), the converter joins two pairs of tensors in each of the 18 linear-attention layers:
  - `attn_qkv` + `attn_gate` → `attn_qkv_gate`;
  - `ssm_beta` + `ssm_alpha` → `ssm_beta_alpha`.
  
  Loaders from b12343 on read both layouts; older ones only the separate one. LM Studio 0.4.28 (engine llama.cpp b12270) refused our file with `tensor 'blk.0.ssm_beta.weight' not found`.

### What changed
1. **Template (`gguf/chat_template.jinja`):** the stock one with two edits:
   - our `SYSTEM` message when the chat has none (a caller's own system message is kept);
   - the empty `<think>\n\n</think>\n\n` block always, whatever `enable_thinking` says.
2. **`general.sampling.temp = 0.0`:** llama.cpp reads it as the default, so it decodes greedily unless the user passes `--temp` or a request sends a temperature.

   Both 1 and 2 were written by `gguf/make_chat_gguf.py`, through gguf-py's own metadata copier. These are the "fused chat" files.
3. **Split (`gguf/split_fused.py`):** each fused tensor is cut back into its two row blocks.
   - The converter built it as plain row concatenation (`torch.cat([first, second], dim=0)`).
   - A Q8_0 or BF16 block never crosses a row, so no byte changes.
   - The file goes from 299 to 335 tensors; metadata is copied unchanged. These are the published files.

### Proofs and results
- **(a) tensors** (`check_chat_gguf.py tensors`, original → fused chat, Q8_0 and BF16): 299 / 299 tensors identical (name, type, shape, sha256 of the bytes). Only `tokenizer.chat_template` changed and `general.sampling.temp` was added.
- **(c) render-offline:** the new template equals `build_prompt()` for 2,980 prompts × {no system, `SYSTEM`} × `enable_thinking` {unset, false, true}: 6 × 2,980 / 2,980. The stock template: 0 / 2,980 without a system message.
- **split-check** (fused chat → split, Q8_0 and BF16):
  - (a) 263 / 263 other tensors byte-identical;
  - (b) 36 / 36 split pairs joined back equal the fused bytes;
  - (c) names, types and shapes without the MTP block equal a reference Qwen3.5-2B GGUF, 320 / 320 (lmstudio-community Q8_0 and unsloth BF16 headers);
  - (d) 44 / 44 metadata keys identical.
- **render-server** (split Q8_0, b12360): PASS, the server's own Jinja engine gives `build_prompt()`.
- **chat** (`/v1/chat/completions`, user message only, greedy, 300 test rows; content must equal the raw `/completion` text of `sft_grpo_gguf_q8`, with empty reasoning):
  - fused chat Q8_0, b12360 Metal: **300 / 300 identical**;
  - split Q8_0, b12360 Metal: **300 / 300 identical**;
  - the original file, stock template: 0 / 300.
- So the scores in this file describe the published files through the chat API as well.

### Compatibility (split Q8_0)

| engine | status | what was run |
|---|---|---|
| llama.cpp b12360, Metal (Mac) | tested | render-server + chat, 300 / 300 identical |
| llama.cpp b12360, CPU in Docker (Step 16c) | tested | system evaluation on 30 rows, on the original file, which has the same tensors; CPU sums in another order, so some near-tied answers differ from Metal |
| llama.cpp b12360, CPU, the GymLeader image (Step 17.2) | tested | the app on 30 test rows with the split Q8_0: greedy text equal to Metal on 19 / 29 model rows, r3 26 / 30 all correct (`system_notes.md`) |
| LM Studio 0.4.28, engine llama.cpp b12270 (Metal) | tested | loads; 3 requests at temperature 0, all `all_correct`, empty reasoning |
| official image `ghcr.io/ggml-org/llama.cpp:server-b12464` (`b12464-d3c3e242c`), CPU | tested | loads; 1 request with no temperature sent, 4 runs identical, `all_correct`, empty reasoning |
| Ollama | not tested | an `hf.co/...` pull picks a Go template from a list and doesn't run our Jinja |

The published BF16 file passed split-check, and through the chat API (llama.cpp b12360, Metal) it gave 30 / 30 answers identical to `sft_grpo_gguf_bf16` on the first 30 test rows.

- **The four answers, scored with `request_parser` + `verify()`:** every request used only spec words (no alias notes).
  - LM Studio, "moderate chest … any equipment … 5 exercises": 5 exercises, all checks pass.
  - LM Studio, "medium forearms … elastic bands": impossible request (2 matching exercises, 4 asked for). The model refused, giving the right count.
  - LM Studio, "hard shoulders … weights": 4 exercises, all checks pass.
  - b12464, "chest workout, medium difficulty, free weights": 4 exercises, all checks pass.
  - All four: strict format, reward 1.0.
- **LM Studio doesn't read `general.sampling.temp`:** set temperature 0 in its settings.
- **What b12464 shows:** with the default temperature of 0.8, four identical answers would be unlikely, so they are consistent with the file's greedy default being used. This is not a proof: one request only.

### sha256

| file | sha256 |
|---|---|
| original BF16 (evaluated above) | `a4767439187a5f6ae01ea3f1ee5c466b600b23c75294fb050f3670dc373fec90` |
| original Q8_0 (evaluated above) | `c1e5904bc7da960affc82b034f7e7d6295fc5659d8cb62017c0fc4dae04b9c74` |
| fused chat BF16 (not published) | `cd8b4531f4a1ed628cfb7ce991b582f4281feb793b22958802707268d61fdb96` |
| fused chat Q8_0 (not published) | `c122afd5683aad2be7720f159ea41cc9518ca363885ddd48e454336c7b2e376d` |
| **published BF16** (chat-ready, split) | `ade25302cea60cf78432b0f92b491e93b7cdcfabed29e70a18d6cd4d8cceb20c` |
| **published Q8_0** (chat-ready, split) | `713efb40bd59dd5dc62b43f2145f540a9780ff6bfcd21006ff36dbceb7df5137` |

## Limits

- **One test set** of 300 requests: enough to separate Q4_K_M from the others, not to rank BF16 and Q8_0, whose differences are within noise.
- **Greedy decoding only.** With sampling the engine and quantization effects might look different.
- **The speeds above were measured on Metal on this Mac.** CPU speeds in Docker are in `system_notes.md`, Docker (CPU).
- The two explanations above (rounding differences between engines, 4-bit rounding blurring GRPO's changes) are **hypotheses, not measured**.

## What's next

- **A smaller file that keeps the GRPO gain:** an importance-matrix (imatrix) quantization, which uses sample text to round the most important weights more carefully, or Q5_K_M / Q6_K. Each would need the same full evaluation against GGUF BF16.
