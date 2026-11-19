---
license: apache-2.0
language: [en]
base_model: Qwen/Qwen3.5-2B
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
tags: [grpo, rl, lora, unsloth, trl, json, fitness]
---

# riccardoroselli/qwen3.5-2b-gym-grpo-lora

The **LoRA adapter** of [`riccardoroselli/qwen3.5-2b-gym-grpo`](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo): Qwen/Qwen3.5-2B trained with SFT, then GRPO, to turn a gym workout request into JSON. For example, "chest workout, medium difficulty, free weights, 4 exercises" gives `{"workout": [{"name", "sets", "reps"}, ...]}` with exercise names from the [exercise catalog](https://github.com/hasaneyldrm/exercises-dataset), or `{"error": "..."}` when not enough exercises match.

**This adapter goes on [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) itself, not on the SFT model.** GRPO kept training the SFT adapter, so this one adapter holds the whole change (SFT, then GRPO). Putting it on top of the SFT model would apply SFT twice.

| | |
|---|---|
| LoRA | r=16, alpha=16, dropout 0, on q/k/v/o/gate/up/down projections; 44 MB |
| base model (`adapter_config.json`) | `Qwen/Qwen3.5-2B` |
| merged with the base | [gym-grpo](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo), the easier way to use the model: its card has MLX code, the full training settings and the val check |

Not tested: loading this adapter with PEFT or mlx-lm. Every evaluation used the merged model or its GGUF files.

## Test results

300 test requests (combinations and phrasings not seen in training, 16 of them impossible), greedy decoding, thinking off, scored by the project's verifier ([`src/verifier.py`](https://github.com/riccardoroselli/SFT-Gym-Workout-Program/blob/main/src/verifier.py)). all_correct = every check passed; reward = the GRPO reward, in [0, 1].

| model | names in catalog | no duplicates | impossible refused | all_correct | reward |
|---|---|---|---|---|---|
| Qwen3.5-2B, zero-shot | – | – | 0/16 | 0.00 | 0.000 |
| Qwen3.5-2B, 3 examples in the prompt | 0.161 | 0.943¹ | 0/16 | 0.00 | 0.415 |
| SFT ([gym-sft](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-sft)) | 0.750 | 0.405 | 4/16 | 0.24 | 0.785 |
| **SFT + GRPO** (this adapter merged: [gym-grpo](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo), MLX bf16) | 0.948 | 0.663 | 6/16 | 0.55 | 0.915 |
| GGUF Q8_0 ([gym-grpo-GGUF](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo-GGUF), llama.cpp) | 0.955 | 0.660 | 6/16 | 0.55 | 0.918 |

¹ High only because few of its names are real catalog names. The zero-shot answers never had the right JSON shape, so their workout columns are empty.

**The GymLeader system, not the model:** the [app](https://github.com/riccardoroselli/SFT-Gym-Workout-Program) wraps the Q8_0 model in a request parser, a guard that answers impossible requests itself, and the verifier, which asks the model again (temperature 0.7) up to `MAX_RETRIES=3` times when an answer fails a check. On the same 300 requests it scores **0.82 all_correct and 0.985 reward, with 16/16 impossible requests answered with an error**. `MAX_RETRIES=3` was fixed before testing, not tuned on the test set, and the system was measured with llama.cpp b12360 on Metal (Apple M4). The model alone is the 0.55 above.

## Known limitations

The same as the merged model's: [gym-grpo, Known limitations](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo#known-limitations). In short:
- only 6 of 16 impossible requests refused;
- about a third of workouts repeat an exercise;
- about 5% of names are not exact catalog names;
- English requests, this one task only;
- one test set, greedy decoding only;
- it suggests workouts; it is not medical advice.

## License

Apache-2.0, the license of the base model [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B). The exercise names come from [hasaneyldrm/exercises-dataset](https://github.com/hasaneyldrm/exercises-dataset) (MIT).

## Links

- Code, data and the full evaluation: [github.com/riccardoroselli/SFT-Gym-Workout-Program](https://github.com/riccardoroselli/SFT-Gym-Workout-Program)
- The models: [gym-sft](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-sft) · [gym-sft-lora](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-sft-lora) · [gym-grpo](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo) · **gym-grpo-lora** (this repo) · [gym-grpo-GGUF](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-grpo-GGUF)
- The app as a Docker image: `ghcr.io/riccardoroselli/gymleader`
