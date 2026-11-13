| run | n | parse_ok | strict_format | names_exist | body_part_ok | equipment_ok | count_ok | no_duplicates | sets_reps_ok | advanced_ok | error_correct | all_correct | reward | has_think | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_zeroshot | 300 | 0.0 | 0.0 | - | - | - | - | - | - | - | 0.0 | 0.0 | 0.0 | 0.0 | 0.14 |
| base_fewshot3 | 300 | 0.9967 | 0.9967 | 0.1611 | 0.0915 | 0.1451 | 0.8829 | 0.9431 | 0.0167 | 1.0 | 0.0 | 0.0 | 0.4147 | 0.0 | 0.0 |
| sft_lora_r16 | 300 | 1.0 | 1.0 | 0.7501 | 0.74 | 0.7305 | 1.0 | 0.4054 | 1.0 | 0.9966 | 0.25 | 0.24 | 0.7846 | 0.0 | 0.0 |
| sft_grpo | 300 | 1.0 | 1.0 | 0.9477 | 0.9338 | 0.9255 | 1.0 | 0.6633 | 0.9966 | 0.9966 | 0.375 | 0.55 | 0.9146 | 0.0 | 0.0 |
| sft_grpo_gguf_bf16 | 300 | 1.0 | 1.0 | 0.9535 | 0.9391 | 0.9312 | 1.0 | 0.6497 | 0.9966 | 0.9966 | 0.375 | 0.54 | 0.9166 | 0.0 | 0.0 |
| sft_grpo_gguf_q8 | 300 | 1.0 | 1.0 | 0.9549 | 0.9404 | 0.9326 | 1.0 | 0.6599 | 0.9966 | 0.9966 | 0.375 | 0.55 | 0.9183 | 0.0 | 0.0 |
| sft_grpo_gguf_q4km | 300 | 1.0 | 1.0 | 0.8152 | 0.803 | 0.7991 | 1.0 | 0.543 | 0.9897 | 0.9931 | 0.4375 | 0.2467 | 0.834 | 0.0 | 0.0 |

All runs are scored with the variant-duplicate rule (docs/spec.md section 2): two names with the same base name, e.g. `x` and `x v. 2`, count as a duplicate.
