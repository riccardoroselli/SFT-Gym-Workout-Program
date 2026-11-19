---
license: apache-2.0
language: [en]
base_model: Qwen/Qwen3.5-2B
base_model_relation: finetune
library_name: transformers
pipeline_tag: text-generation
new_version: riccardoroselli/qwen3.5-2b-gym-grpo
tags: [sft, lora, unsloth, trl, json, fitness]
---

# riccardoroselli/qwen3.5-2b-gym-sft

> **This is the SFT stage, not the final model.** The final model is [`riccardoroselli/qwen3.5-2b-gym-grpo`](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo): this model trained further with GRPO, which raises all_correct from 0.24 to 0.55 on the test set.

Qwen/Qwen3.5-2B fine-tuned with SFT (LoRA) to turn a gym workout request like "chest workout, medium difficulty, free weights" into JSON: `{"workout": [{"name", "sets", "reps"}, ...]}` with exercise names from the [exercise catalog](https://github.com/hasaneyldrm/exercises-dataset), or `{"error": "..."}` when not enough exercises match the request. This repo holds the **merged 16-bit model** (base + LoRA, bf16).

- Code: [github.com/riccardoroselli/SFT-Gym-Workout-Program](https://github.com/riccardoroselli/SFT-Gym-Workout-Program), trained with [`notebooks/02_sft_colab.ipynb`](https://github.com/riccardoroselli/SFT-Gym-Workout-Program/blob/main/notebooks/02_sft_colab.ipynb)
- Data: tag [`data-v1`](https://github.com/riccardoroselli/SFT-Gym-Workout-Program/tree/data-v1/data/processed): 2,000 train rows, 200 val rows. The test set was not used for training.
- Prompt: the dataset's system message + the request, with Qwen's chat template and thinking off. The MLX code on the [gym-grpo card](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo) works for this model too, with this repo's name.

## Test results

300 test requests (combinations and phrasings not seen in training, 16 of them impossible), greedy decoding, thinking off, scored by the project's verifier ([`src/verifier.py`](https://github.com/riccardoroselli/SFT-Gym-Workout-Program/blob/main/src/verifier.py)). all_correct = every check passed; reward = the GRPO reward, in [0, 1].

| model | names in catalog | no duplicates | impossible refused | all_correct | reward |
|---|---|---|---|---|---|
| Qwen3.5-2B, zero-shot | – | – | 0/16 | 0.00 | 0.000 |
| Qwen3.5-2B, 3 examples in the prompt | 0.161 | 0.943¹ | 0/16 | 0.00 | 0.415 |
| **SFT** (this model, MLX bf16) | 0.750 | 0.405 | 4/16 | 0.24 | 0.785 |
| SFT + GRPO ([gym-grpo](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo), MLX bf16) | 0.948 | 0.663 | 6/16 | 0.55 | 0.915 |
| GGUF Q8_0 ([gym-grpo-GGUF](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo-GGUF), llama.cpp) | 0.955 | 0.660 | 6/16 | 0.55 | 0.918 |

¹ High only because few of its names are real catalog names. The zero-shot answers never had the right JSON shape, so their workout columns are empty.

**The GymLeader system, not the model:** the [app](https://github.com/riccardoroselli/SFT-Gym-Workout-Program) wraps the Q8_0 model in a request parser, a guard that answers impossible requests itself, and the verifier, which asks the model again (temperature 0.7) up to `MAX_RETRIES=3` times when an answer fails a check. On the same 300 requests it scores **0.82 all_correct and 0.985 reward, with 16/16 impossible requests answered with an error**. `MAX_RETRIES=3` was fixed before testing, not tuned on the test set, and the system was measured with llama.cpp b12360 on Metal (Apple M4). The model alone is the 0.55 above.

SFT taught the format, the count and the sets/reps (every test answer has the right JSON shape and the right sets/reps). Its failures are exact names and duplicates. Details: [`eval/sft_notes.md`](https://github.com/riccardoroselli/SFT-Gym-Workout-Program/blob/main/eval/sft_notes.md).

## Training

| setting | value |
|---|---|
| method | bf16 LoRA (not QLoRA) with Unsloth + TRL `SFTTrainer`, loss on the answer only |
| LoRA | r=16, alpha=16, dropout 0, on q/k/v/o/gate/up/down projections |
| learning rate | 0.0002, cosine, 10 warmup steps, 8-bit AdamW, weight decay 0.01 |
| batch | 8 × 2 gradient accumulation = 16 rows per step |
| epochs / steps | 2 / 250 |
| max length / seed | 1024 / 42 |
| GPU | Tesla T4 |

![train and val loss](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-sft/resolve/main/loss_curves.png)

Final train loss **0.2562**, final val loss **0.2607**.

## Quick check during training

30 val rows (not test), greedy decoding, max 512 new tokens, scored with the project's `verify()`.

| check | mean |
|---|---|
| parse_ok | 1.0 |
| strict_format | 1.0 |
| names_exist | 0.8822 |
| body_part_ok | 0.8822 |
| equipment_ok | 0.8822 |
| count_ok | 1.0 |
| no_duplicates | 0.6 |
| sets_reps_ok | 1.0 |
| advanced_ok | 1.0 |
| error_correct | - |
| all_correct | 0.4333 |
| reward | 0.9004 |
| has_think | 0.0 |
| truncated | 0.0 |

## Known limitations

- **Duplicates:** 59% of test workouts repeat an exercise or list two variants of it (no_duplicates 0.41). GRPO brought this down to 34%.
- **Names:** 74% of listed exercises are exact catalog names; the rest are near-misses (`dumbbell incline chest press` for `dumbbell incline bench press`) or invented.
- **Impossible requests:** only 4 of 16 refused. The training data's impossible requests come from only 5 body part × equipment pairs, so the model may have learned those pairs by heart rather than to count matching exercises.
- English requests, this one task only: it is not a general chat model.
- Evaluated on one test set of 300 requests, with greedy decoding only.
- It suggests workouts; it is not medical advice.

## License

Apache-2.0, the license of the base model [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B). The exercise names come from [hasaneyldrm/exercises-dataset](https://github.com/hasaneyldrm/exercises-dataset) (MIT).

## Links

- Code, data and the full evaluation: [github.com/riccardoroselli/SFT-Gym-Workout-Program](https://github.com/riccardoroselli/SFT-Gym-Workout-Program)
- The models: **gym-sft** (this repo) · [gym-sft-lora](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-sft-lora) · [gym-grpo](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo) · [gym-grpo-lora](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo-lora) · [gym-grpo-GGUF](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo-GGUF)
- The app as a Docker image: `ghcr.io/riccardoroselli/gymleader`
