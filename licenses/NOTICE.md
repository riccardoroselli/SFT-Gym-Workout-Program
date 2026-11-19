# What GymLeader bundles, and under which license

| Part | Where | License | Text |
|---|---|---|---|
| GymLeader code, web page and data-v1 | this repository and the image | Apache-2.0 | `LICENSE` (in the image: `/licenses/gymleader/LICENSE`) |
| Exercise catalog (`data/processed/catalog.json`), derived from [exercises-dataset](https://github.com/hasaneyldrm/exercises-dataset) at commit `7455efa` | this repository and the image | MIT, © 2026 Hasan Emir Yıldırım. Only the data is used: none of the dataset's media (© Gym visual), which the MIT license does not cover | `licenses/exercises-dataset-MIT.txt` |
| llama.cpp `llama-server` and its libraries, tag b12360 (commit `2d8a32502`) | the Docker image only, `/opt/llama` | MIT, © The ggml authors; bundled third-party code under `licenses/` | `/licenses/llama.cpp/` (in the image) |
| Qwen3.5-2B tokenizer files (`Qwen/Qwen3.5-2B` at revision `15852e8c`) | the Docker image only, `/app/tokenizer` | Apache-2.0, Qwen team (Alibaba Cloud) | `/licenses/qwen-tokenizer/LICENSE` (in the image) |
| Python packages (FastAPI, uvicorn, pydantic, transformers, tokenizers, huggingface_hub, …) | the Docker image only | their own licenses | each package's `*.dist-info/` in `/usr/local/lib/python3.12/site-packages/` |

The model weights are not in the image: on the first start the container downloads them from
`riccardoroselli/qwen3.5-2b-gym-grpo-GGUF` (Apache-2.0, fine-tuned from Qwen/Qwen3.5-2B, Apache-2.0).

GymLeader suggests workouts; it is not medical advice.
