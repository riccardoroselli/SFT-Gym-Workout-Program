# GymLeader API (src/api.py). Build context: the repo root (see docker-compose.dev.yml).
# The same base as docker/Dockerfile, pinned to the multi-arch index (linux/amd64 + linux/arm64).
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# The tokenizer files only, from the public base model at a pinned revision, no token. Checked in eval/system_notes.md ("Docker (CPU)"):
# build_prompt() gives the same text and token IDs as riccardoroselli/qwen3.5-2b-gym-grpo on all 2,500 data-v1
# requests and the 480 canonical sentences. config.json makes transformers load the same tokenizer class.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-2B', \
    revision='15852e8c16360a2fea060d615a32b45270f8a8fc', token=False, local_dir='/app/tokenizer', \
    allow_patterns=['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'config.json'])" \
    && rm -rf /app/tokenizer/.cache

# api.py reads only src/ and the catalog (verifier.py: ROOT / data/processed/catalog.json)
COPY src/ src/
COPY data/processed/catalog.json data/processed/catalog.json

RUN useradd --create-home --uid 1000 app
USER app

ENV HF_HUB_OFFLINE=1 \
    TOKENIZER=/app/tokenizer \
    PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
