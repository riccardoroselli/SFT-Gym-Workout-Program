---
license: apache-2.0
language: [en]
base_model: Qwen/Qwen3.5-2B
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
new_version: riccardoroselli/qwen3.5-2b-gym-grpo-lora
tags: [sft, lora, unsloth, trl, json, fitness]
---

# riccardoroselli/qwen3.5-2b-gym-sft-lora

> **This is the SFT stage, not the final model.** The final adapter is [`riccardoroselli/qwen3.5-2b-gym-grpo-lora`](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo-lora): this adapter trained further with GRPO, which raises all_correct from 0.24 to 0.55 on the test set.

The **LoRA adapter** of [`riccardoroselli/qwen3.5-2b-gym-sft`](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-sft): Qwen/Qwen3.5-2B fine-tuned with SFT to turn a gym workout request into JSON, `{"workout": [{"name", "sets", "reps"}, ...]}` with exercise names from the [exercise catalog](https://github.com/hasaneyldrm/exercises-dataset), or `{"error": "..."}` when not enough exercises match.

| | |
|---|---|
| LoRA | r=16, alpha=16, dropout 0, on q/k/v/o/gate/up/down projections; 44 MB |
| base model (`adapter_config.json`) | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) |
| merged with the base | [gym-sft](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-sft): its card has the training settings, the loss curves and the val check |

GRPO training started from this adapter: [`notebooks/03_grpo_colab.ipynb`](https://github.com/riccardoroselli/SFT-Gym-Workout-Program/blob/main/notebooks/03_grpo_colab.ipynb) loads it with Unsloth on Colab. Not tested: loading it with PEFT or mlx-lm on a Mac. Every evaluation used the merged model.

## Test results

300 test requests (combinations and phrasings not seen in training, 16 of them impossible), greedy decoding, thinking off, scored by the project's verifier ([`src/verifier.py`](https://github.com/riccardoroselli/SFT-Gym-Workout-Program/blob/main/src/verifier.py)). all_correct = every check passed; reward = the GRPO reward, in [0, 1].

| model | names in catalog | no duplicates | impossible refused | all_correct | reward |
|---|---|---|---|---|---|
| Qwen3.5-2B, zero-shot | – | – | 0/16 | 0.00 | 0.000 |
| Qwen3.5-2B, 3 examples in the prompt | 0.161 | 0.943¹ | 0/16 | 0.00 | 0.415 |
| **SFT** (this adapter merged: [gym-sft](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-sft), MLX bf16) | 0.750 | 0.405 | 4/16 | 0.24 | 0.785 |
| SFT + GRPO ([gym-grpo](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo), MLX bf16) | 0.948 | 0.663 | 6/16 | 0.55 | 0.915 |
| GGUF Q8_0 ([gym-grpo-GGUF](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo-GGUF), llama.cpp) | 0.955 | 0.660 | 6/16 | 0.55 | 0.918 |

¹ High only because few of its names are real catalog names. The zero-shot answers never had the right JSON shape, so their workout columns are empty.

**The GymLeader system, not the model:** the [app](https://github.com/riccardoroselli/SFT-Gym-Workout-Program) wraps the Q8_0 model in a request parser, a guard that answers impossible requests itself, and the verifier, which asks the model again (temperature 0.7) up to `MAX_RETRIES=3` times when an answer fails a check. On the same 300 requests it scores **0.82 all_correct and 0.985 reward, with 16/16 impossible requests answered with an error**. `MAX_RETRIES=3` was fixed before testing, not tuned on the test set, and the system was measured with llama.cpp b12360 on Metal (Apple M4). The model alone is the 0.55 above.

## Known limitations

The same as the merged model's: [gym-sft, Known limitations](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-sft#known-limitations). In short:
- 59% of workouts repeat an exercise;
- 74% of listed names are exact catalog names;
- only 4 of 16 impossible requests refused;
- English requests, this one task only;
- one test set, greedy decoding only;
- it suggests workouts; it is not medical advice.

## License

Apache-2.0, the license of the base model [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B). The exercise names come from [hasaneyldrm/exercises-dataset](https://github.com/hasaneyldrm/exercises-dataset) (MIT).

## Links

- Code, data and the full evaluation: [github.com/riccardoroselli/SFT-Gym-Workout-Program](https://github.com/riccardoroselli/SFT-Gym-Workout-Program)
- The models: [gym-sft](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-sft) · **gym-sft-lora** (this repo) · [gym-grpo](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo) · [gym-grpo-lora](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo-lora) · [gym-grpo-GGUF](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo-GGUF)
- The app as a Docker image: `ghcr.io/riccardoroselli/gymleader`
