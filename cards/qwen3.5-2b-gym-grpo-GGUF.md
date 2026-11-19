---
license: apache-2.0
language: [en]
base_model: riccardoroselli/qwen3.5-2b-gym-grpo
base_model_relation: quantized
pipeline_tag: text-generation
tags: [gguf, llama.cpp, grpo, json, fitness]
---

# riccardoroselli/qwen3.5-2b-gym-grpo-GGUF

GGUF files of [`riccardoroselli/qwen3.5-2b-gym-grpo`](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo), for llama.cpp, LM Studio and other GGUF apps. The model is Qwen/Qwen3.5-2B fine-tuned with SFT (LoRA), then GRPO, to turn a gym workout request into JSON:

```text
request: chest workout, medium difficulty, free weights, 4 exercises
answer:  {"workout":[{"name":"dumbbell incline fly","sets":3,"reps":"8-12"},{"name":"dumbbell incline fly on exercise ball","sets":3,"reps":"8-12"},{"name":"dumbbell incline bench press","sets":3,"reps":"8-12"},{"name":"dumbbell fly","sets":3,"reps":"8-12"}]}
```

The answer is `{"workout": [{"name", "sets", "reps"}, ...]}` with exercise names from the [exercise catalog](https://github.com/hasaneyldrm/exercises-dataset), or `{"error": "..."}` when not enough exercises match the request.

**The files are chat-ready.** Their chat template adds the training system message when a chat has none and always turns thinking off, and their default temperature is 0. A plain chat request gets the same prompt the model was trained and evaluated with.

## Files

At revision [`bbb1d3cf7f9113ec518b56c95a74702bbd83ba9b`](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo-GGUF/tree/bbb1d3cf7f9113ec518b56c95a74702bbd83ba9b):

| file | size | sha256 | use |
|---|---|---|---|
| `qwen3.5-2b-gym-grpo-Q8_0.gguf` | 2,076,675,008 bytes (8.5 bits per weight) | `713efb40bd59dd5dc62b43f2145f540a9780ff6bfcd21006ff36dbceb7df5137` | **recommended** |
| `qwen3.5-2b-gym-grpo-BF16.gguf` | 3,897,387,968 bytes (16 bits per weight) | `ade25302cea60cf78432b0f92b491e93b7cdcfabed29e70a18d6cd4d8cceb20c` | reference |

| file | writing, tg128 | reading the prompt, pp512 |
|---|---|---|
| Q8_0 | 43.2 tok/s | 918 tok/s |
| BF16 | 25.5 tok/s | 958 tok/s |

Speeds from `llama-bench` (llama.cpp b12360) on an Apple M4 MacBook Air: Metal, all layers on the GPU, 3 repetitions. They will be different on other hardware, and lower on a CPU.

**Q4_K_M is not published.** It was measured, and it loses most of what GRPO added: all_correct 0.25 against 0.54 for BF16, and names in the catalog 0.82 against 0.95.

## How to use

### LM Studio

Search for `riccardoroselli/qwen3.5-2b-gym-grpo-GGUF`, download the Q8_0 file, and set **Temperature 0** in the model's settings: LM Studio doesn't read the file's default temperature. Then write a request in the chat.

Tested with LM Studio 0.4.28 (engine llama.cpp b12270), loading a local copy of the file: 3 requests, all correct, empty reasoning.

### llama-server, chat API

```bash
hf download riccardoroselli/qwen3.5-2b-gym-grpo-GGUF qwen3.5-2b-gym-grpo-Q8_0.gguf \
    --revision bbb1d3cf7f9113ec518b56c95a74702bbd83ba9b --local-dir .
llama-server -m qwen3.5-2b-gym-grpo-Q8_0.gguf -ngl all -c 4096 -np 1 --host 127.0.0.1 --port 8080
```

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "chest workout, medium difficulty, free weights, 4 exercises"}],
  "temperature": 0, "max_tokens": 512}'
```

The answer is in `choices[0].message.content`. Through this API the Q8_0 file gave exactly the evaluated answer on all 300 test requests (llama.cpp b12360, Metal), with no system message sent.

### llama-server, raw prompt (how it was evaluated)

The test results below come from this request: the exact training prompt, sent to `/completion`, so the server's chat template isn't used.

```bash
curl -s http://127.0.0.1:8080/completion -H "Content-Type: application/json" -d @- <<'EOF'
{"prompt": "<|im_start|>system\nYou create gym workouts from the user's request. Answer only with JSON.<|im_end|>\n<|im_start|>user\nchest workout, medium difficulty, free weights, 4 exercises<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
 "n_predict": 512, "temperature": 0, "stop": ["<|im_end|>"], "cache_prompt": false,
 "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "dry_multiplier": 0.0}
EOF
```

The answer is in the `content` field.

### The GymLeader app (Docker)

```bash
docker run -d --name gymleader -p 127.0.0.1:8000:8000 -v gymleader-models:/models ghcr.io/riccardoroselli/gymleader:v1.0
```

Then open http://127.0.0.1:8000. The image (linux/amd64 and linux/arm64, CPU) downloads the Q8_0 file from this repo at the revision above on its first start and checks its sha256. It runs the whole GymLeader system, with the guard and the verifier retries described under the results.

### MLX (Apple silicon)

Use the safetensors model [`riccardoroselli/qwen3.5-2b-gym-grpo`](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo); its card has the code.

## Recommended settings

| setting | value | why |
|---|---|---|
| temperature | 0 | greedy decoding, as evaluated. It is the file's default; LM Studio needs it set by hand |
| max tokens | 512 | the longest test answer was 157 tokens |
| stop | `<\|im_end\|>` | only for raw `/completion`; the chat API stops there by itself |
| system message | none | the template adds the training one. A system message of your own replaces it, and wasn't evaluated |
| thinking | off | built into the template: the model never saw a thinking block in training |
| repetition penalties | off | llama.cpp's default; every evaluation ran with them off |

## Test results

300 test requests (combinations and phrasings not seen in training, 16 of them impossible), greedy decoding, thinking off, scored by the project's verifier ([`src/verifier.py`](https://github.com/riccardoroselli/SFT-Gym-Workout-Program/blob/main/src/verifier.py)). all_correct = every check passed; reward = the GRPO reward, in [0, 1].

| model | names in catalog | no duplicates | impossible refused | all_correct | reward |
|---|---|---|---|---|---|
| Qwen3.5-2B, zero-shot | – | – | 0/16 | 0.00 | 0.000 |
| Qwen3.5-2B, 3 examples in the prompt | 0.161 | 0.943¹ | 0/16 | 0.00 | 0.415 |
| SFT ([gym-sft](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-sft)) | 0.750 | 0.405 | 4/16 | 0.24 | 0.785 |
| SFT + GRPO ([gym-grpo](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo), MLX bf16) | 0.948 | 0.663 | 6/16 | 0.55 | 0.915 |
| **GGUF Q8_0** (this repo, llama.cpp) | 0.955 | 0.660 | 6/16 | 0.55 | 0.918 |

¹ High only because few of its names are real catalog names. The zero-shot answers never had the right JSON shape, so their workout columns are empty.

**The GymLeader system, not the model:** the [app](https://github.com/riccardoroselli/SFT-Gym-Workout-Program) wraps the Q8_0 model in a request parser, a guard that answers impossible requests itself, and the verifier, which asks the model again (temperature 0.7) up to `MAX_RETRIES=3` times when an answer fails a check. On the same 300 requests it scores **0.82 all_correct and 0.985 reward, with 16/16 impossible requests answered with an error**. `MAX_RETRIES=3` was fixed before testing, not tuned on the test set, and the system was measured with llama.cpp b12360 on Metal (Apple M4). The model alone is the 0.55 above.

MLX bf16, GGUF BF16 and GGUF Q8_0 score the same within noise: in a paired bootstrap on the same 300 requests, the 95% intervals of MLX → BF16 and BF16 → Q8_0 contain 0, for both reward and all_correct. Details: [`eval/gguf_notes.md`](https://github.com/riccardoroselli/SFT-Gym-Workout-Program/blob/main/eval/gguf_notes.md).

## Compatibility (Q8_0)

| engine | status | what was run |
|---|---|---|
| llama.cpp b12360, Metal (Mac) | tested | the chat API on the 300 test requests: 300/300 identical to the evaluated answers |
| llama.cpp b12360, CPU (the GymLeader Docker image) | tested | the app on 30 test requests: 26/30 all correct with retries. Greedy answers equal Metal's on 19 of 29: the CPU adds numbers in another order, which can flip a near-tie |
| LM Studio 0.4.28, engine llama.cpp b12270 (Metal) | tested | loads; 3 requests at temperature 0, all correct, empty reasoning |
| llama.cpp b12464, official image `ghcr.io/ggml-org/llama.cpp:server-b12464`, CPU | tested | loads; 1 request with no temperature sent, 4 runs identical, all correct, empty reasoning |
| Ollama | not tested | it picks a Go template from a list instead of running the file's Jinja template, so the prompt may differ |

The BF16 file was checked through the chat API on 30 test requests (llama.cpp b12360, Metal): 30/30 identical to its evaluated answers.

## What's in the files

- **Conversion:** llama.cpp b12360 (commit `2d8a32502`), `convert_hf_to_gguf.py` → BF16, then `llama-quantize` → Q8_0 (norms stay F32). Architecture `qwen35`, 1.94 B parameters. The tokenizer gives the same token IDs as the Hugging Face one on the evaluation prompt.
- **Text only:** Qwen3.5's vision tower is not included (there is no mmproj file).
- **The multi-token-prediction (MTP) layer** is kept (block 24) but not used. llama.cpp prints 15 `unused tensor blk.24…` warnings when it loads the file; they are harmless.
- **Split tensors, for older loaders:** since b12343, llama.cpp's converter joins two pairs of tensors in each of the 18 linear-attention layers (`attn_qkv` + `attn_gate`, `ssm_beta` + `ssm_alpha`). Older loaders, such as LM Studio's b12270, can't read that. These files store the pairs separately again (335 tensors instead of 299), which old and new loaders both read. The split only cuts between rows: the bytes are the same as in the joined file.
- **Chat template:** Qwen3.5's own, with two changes: the training system message when the chat has none, and the empty `<think>\n\n</think>\n\n` block always. `general.sampling.temp = 0.0` sets greedy decoding as the default. The weights are the evaluated ones: only the template, the default temperature and the tensor split changed.

## Known limitations

- **Impossible requests:** the model refuses only 6 of 16 and often writes a workout instead (the app's guard catches all 16). The training data's impossible requests come from only 5 body part × equipment pairs (forearms + bands, forearms + bodyweight, shoulders + bodyweight, calves + bands, chest + bands), so the model may have learned those pairs by heart rather than to count matching exercises. Its refusals may not carry over if the catalog changes.
- **Duplicates:** about a third of workouts repeat an exercise, or list two variants of it, like `x` and `x v. 2`.
- **Names:** about 5% of listed exercises are not exact catalog names, and some wrong names come back again and again (`bodyweight standing archer`).
- English requests, this one task only: it is not a general chat model.
- Evaluated on one test set of 300 requests, with greedy decoding only.
- It suggests workouts; it is not medical advice.

## License

Apache-2.0, the license of the base model [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B). The exercise names come from [hasaneyldrm/exercises-dataset](https://github.com/hasaneyldrm/exercises-dataset) (MIT).

## Links

- Code, data and the full evaluation: [github.com/riccardoroselli/SFT-Gym-Workout-Program](https://github.com/riccardoroselli/SFT-Gym-Workout-Program)
- The models: [gym-sft](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-sft) · [gym-sft-lora](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-sft-lora) · [gym-grpo](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo) · [gym-grpo-lora](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo-lora) · **gym-grpo-GGUF** (this repo)
- The app as a Docker image: `ghcr.io/riccardoroselli/gymleader`
