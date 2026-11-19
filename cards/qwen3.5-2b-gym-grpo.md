---
license: apache-2.0
language: [en]
base_model: riccardoroselli/qwen3.5-2b-gym-sft
base_model_relation: finetune
library_name: transformers
pipeline_tag: text-generation
tags: [grpo, rl, lora, unsloth, trl, json, fitness]
---

# riccardoroselli/qwen3.5-2b-gym-grpo

Qwen/Qwen3.5-2B trained in two stages to turn a gym workout request into JSON: SFT ([gym-sft](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-sft)), then **GRPO** (this model), which kept training the SFT LoRA adapter with the project's verifier as the reward.

```text
request: chest workout, medium difficulty, free weights, 4 exercises
answer:  {"workout":[{"name":"dumbbell incline fly","sets":3,"reps":"8-12"},{"name":"dumbbell incline fly on exercise ball","sets":3,"reps":"8-12"},{"name":"dumbbell incline bench press","sets":3,"reps":"8-12"},{"name":"dumbbell fly","sets":3,"reps":"8-12"}]}
```

The answer is `{"workout": [{"name", "sets", "reps"}, ...]}` with exercise names from the [exercise catalog](https://github.com/hasaneyldrm/exercises-dataset), or `{"error": "..."}` when not enough exercises match the request.

This repo holds the **merged 16-bit model** (base + LoRA, bf16). This is the final model. The same model is also available:
- as GGUF files, for llama.cpp and LM Studio: [gym-grpo-GGUF](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo-GGUF);
- as a LoRA adapter only: [gym-grpo-lora](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo-lora).

- Code: [github.com/riccardoroselli/SFT-Gym-Workout-Program](https://github.com/riccardoroselli/SFT-Gym-Workout-Program), trained with [`notebooks/03_grpo_colab.ipynb`](https://github.com/riccardoroselli/SFT-Gym-Workout-Program/blob/main/notebooks/03_grpo_colab.ipynb)
- Data: tag [`data-v1`](https://github.com/riccardoroselli/SFT-Gym-Workout-Program/tree/data-v1/data/processed). GRPO used 600 train prompts, 20% of them impossible requests. The test set was not used for training.

## How to use (MLX, Apple silicon)

```python
from mlx_lm import load, generate

model, tokenizer = load("riccardoroselli/qwen3.5-2b-gym-grpo")
messages = [
    {"role": "system", "content": "You create gym workouts from the user's request. Answer only with JSON."},
    {"role": "user", "content": "chest workout, medium difficulty, free weights, 4 exercises"},
]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
print(generate(model, tokenizer, prompt, max_tokens=512))  # greedy: mlx-lm's default sampler
```

Tested with mlx-lm 0.32.2. It gives the answer shown above, and on the first 3 test requests it gives exactly the evaluated answers.

- **Keep the system message and `enable_thinking=False`.** This repo's chat template is Qwen3.5's own, so, unlike the GGUF files, it doesn't add them for you. Every training example had this system message and no thinking block.
- **Use greedy decoding** (temperature 0), as in every evaluation.
- Not tested: Transformers, vLLM, and image input. Qwen3.5's vision part is still in the weights, but only text was trained and evaluated.

## Test results

300 test requests (combinations and phrasings not seen in training, 16 of them impossible), greedy decoding, thinking off, scored by the project's verifier ([`src/verifier.py`](https://github.com/riccardoroselli/SFT-Gym-Workout-Program/blob/main/src/verifier.py)). all_correct = every check passed; reward = the GRPO reward, in [0, 1].

| model | names in catalog | no duplicates | impossible refused | all_correct | reward |
|---|---|---|---|---|---|
| Qwen3.5-2B, zero-shot | – | – | 0/16 | 0.00 | 0.000 |
| Qwen3.5-2B, 3 examples in the prompt | 0.161 | 0.943¹ | 0/16 | 0.00 | 0.415 |
| SFT ([gym-sft](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-sft)) | 0.750 | 0.405 | 4/16 | 0.24 | 0.785 |
| **SFT + GRPO** (this model, MLX bf16) | 0.948 | 0.663 | 6/16 | 0.55 | 0.915 |
| GGUF Q8_0 ([gym-grpo-GGUF](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo-GGUF), llama.cpp) | 0.955 | 0.660 | 6/16 | 0.55 | 0.918 |

¹ High only because few of its names are real catalog names. The zero-shot answers never had the right JSON shape, so their workout columns are empty.

**The GymLeader system, not the model:** the [app](https://github.com/riccardoroselli/SFT-Gym-Workout-Program) wraps the Q8_0 model in a request parser, a guard that answers impossible requests itself, and the verifier, which asks the model again (temperature 0.7) up to `MAX_RETRIES=3` times when an answer fails a check. On the same 300 requests it scores **0.82 all_correct and 0.985 reward, with 16/16 impossible requests answered with an error**. `MAX_RETRIES=3` was fixed before testing, not tuned on the test set, and the system was measured with llama.cpp b12360 on Metal (Apple M4). The model alone is the 0.55 above.

GRPO more than doubled all_correct (0.24 → 0.55), so the reward gain comes from more fully correct workouts, not only from partial credit. Details: [`eval/grpo_notes.md`](https://github.com/riccardoroselli/SFT-Gym-Workout-Program/blob/main/eval/grpo_notes.md).

## Reward

The reward is the project's verifier, `verify(request, answer)["reward"]` from `src/verifier.py`, in [0, 1]:
- 10% for the exact JSON format;
- 90% for the content: names in the catalog 0.30, body part 0.15, equipment 0.15, then 0.10 each for the count, no duplicates, sets/reps, and no advanced exercise in easy or medium workouts.

An impossible request scores 1 only for an error answer. Names that differ only by "v. N", "(male)" or "(female)" are the same exercise.

## Training

| setting | value |
|---|---|
| method | GRPO with TRL `GRPOTrainer` + Unsloth, continuing the SFT LoRA (r=16, alpha=16) |
| answers per prompt / prompts per step | 8 / 2 |
| sampling temperature / max new tokens | 0.7 / 200 |
| KL penalty (beta) | 0.0 |
| learning rate | 5e-06, constant after 10 warmup steps; 8-bit AdamW (0.9, 0.99), weight decay 0.1, max grad norm 0.1 |
| steps | 300 (max 300, time budget 100 min) |
| libraries / seed | trl 1.15.0, unsloth 2026.11.4; other GRPO settings at TRL's defaults / 42 |
| GPU | Tesla T4 |

![GRPO curves](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo/resolve/main/grpo_curves.png)

## Val check during training

40 val rows (13 impossible + 27 feasible), greedy decoding, scored with `verify()`. Step 0 is the starting model.

| check | step 0 | final |
|---|---|---|
| parse_ok | 1.0 | 1.0 |
| strict_format | 1.0 | 1.0 |
| names_exist | 0.8495 | 0.9185 |
| body_part_ok | 0.827 | 0.8914 |
| equipment_ok | 0.7685 | 0.8329 |
| count_ok | 1.0 | 1.0 |
| no_duplicates | 0.3784 | 0.6216 |
| sets_reps_ok | 1.0 | 1.0 |
| advanced_ok | 1.0 | 1.0 |
| error_correct | 0.2308 | 0.2308 |
| all_correct | 0.225 | 0.45 |
| reward | 0.6778 | 0.7408 |
| duplicate_rate | 0.6216 | 0.3784 |
| refusal_rate | 0.2308 | 0.2308 |
| false_refusal_rate | 0.0 | 0.0 |
| truncated | 0.0 | 0.0 |

## Known limitations

- **Impossible requests:** the model refuses only 6 of 16 and often writes a workout instead (the app's guard catches all 16). The training data's impossible requests come from only 5 body part × equipment pairs (forearms + bands, forearms + bodyweight, shoulders + bodyweight, calves + bands, chest + bands), so the model may have learned those pairs by heart rather than to count matching exercises. Its refusals may not carry over if the catalog changes.
- **Little or no refusal signal on some pairs during GRPO.** Before training, the SFT model answered 8 impossible train requests 8 times each at temperature 0.7, the training temperature:
  - **chest / bands** never refused (0 of 8), so GRPO had no refusal to reward;
  - **shoulders / bodyweight** refused 1 of 8, a weak signal;
  - **forearms / bodyweight** refused 0 of 8 on one request (easy, 3 exercises) but 2 of 8 on the other (hard, 6).
- **Duplicates:** about a third of workouts repeat an exercise, or list two variants of it, like `x` and `x v. 2`.
- **Names:** about 5% of listed exercises are not exact catalog names, and some wrong names come back again and again (`bodyweight standing archer`).
- English requests, this one task only: it is not a general chat model.
- Evaluated on one test set of 300 requests, with greedy decoding only.
- It suggests workouts; it is not medical advice.

## License

Apache-2.0, the license of the base model [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B). The exercise names come from [hasaneyldrm/exercises-dataset](https://github.com/hasaneyldrm/exercises-dataset) (MIT).

## Links

- Code, data and the full evaluation: [github.com/riccardoroselli/SFT-Gym-Workout-Program](https://github.com/riccardoroselli/SFT-Gym-Workout-Program)
- The models: [gym-sft](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-sft) · [gym-sft-lora](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-sft-lora) · **gym-grpo** (this repo) · [gym-grpo-lora](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo-lora) · [gym-grpo-GGUF](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo-GGUF)
- The app as a Docker image: `ghcr.io/riccardoroselli/gymleader`
