<img src="img/GymLeader-Banner.jpg" alt="GymLeader banner" width="100%">

# 🏋🏻‍♂️ GymLeader: Supervised Fine-Tuning of Qwen3.5-2B to create personalized Gym Workout Plans

Work In Progress ... 😎

## Model weights

SFT checkpoint (not the final model: GRPO comes next), on the Hugging Face Hub:

| Model | Hugging Face | Contents |
|---|---|---|
| SFT, merged | [riccardoroselli/qwen3.5-2b-gym-sft](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-sft) | Qwen3.5-2B + LoRA merged into one bf16 model, with training logs and loss curves |
| SFT, LoRA adapter | [riccardoroselli/qwen3.5-2b-gym-sft-lora](https://huggingface.co/riccardoroselli/qwen3.5-2b-gym-sft-lora) | the LoRA adapter only, to load on top of [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) |

Trained with [`notebooks/02_sft_colab.ipynb`](notebooks/02_sft_colab.ipynb) (executed, with outputs).
