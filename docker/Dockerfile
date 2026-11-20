# syntax=docker/dockerfile:1
# GymLeader, all in one image: llama-server (llama.cpp b12360, CPU) + the API + the web page. The model is not in
# the image: src/serve.py downloads it into /models on the first start (pinned revision, sha256-checked).
#   docker build -f docker/Dockerfile -t gymleader .
#   docker run -p 127.0.0.1:8000:8000 -v gymleader-models:/models gymleader     ->  http://127.0.0.1:8000
# Both stages use the same base, so llama-server is linked against the glibc it runs with.
ARG BASE=python:3.12-slim

FROM ${BASE} AS llama-build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*
# The tag every GGUF eval used; --checksum makes the build fail if the tag ever points to another commit.
ADD --checksum=2d8a32502909ca065c41c4cf5f2fb1c427c047fc https://github.com/ggml-org/llama.cpp.git#b12360 /src
WORKDIR /src
# Compute flags as upstream .devops/cpu.Dockerfile (one library per CPU generation, the best is picked at run time).
# The build number and commit are given, since there is no .git here; with them, --version and /props say
# b12360-2d8a32502 as on the Mac. No llama.cpp web UI (it would be downloaded at build time) and no OpenSSL.
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF \
        -DLLAMA_BUILD_NUMBER=12360 -DLLAMA_BUILD_COMMIT=2d8a32502 \
    && cmake --build build -j"$(nproc)" --target llama-server \
    && mkdir -p /out/llama /out/licenses/llama.cpp \
    && cp build/bin/llama-server /out/llama/ \
    && find build -name "*.so*" -exec cp -P {} /out/llama/ \; \
    && cp -r LICENSE licenses /out/licenses/llama.cpp/

FROM ${BASE}
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# The tokenizer files only (+ their license), from the public base model at a pinned revision, no token.
# They give the same prompts and token IDs as riccardoroselli/qwen3.5-2b-gym-grpo (eval/system_notes.md, "Docker (CPU)").
# serve.py has the same list.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-2B', \
    revision='15852e8c16360a2fea060d615a32b45270f8a8fc', token=False, local_dir='/app/tokenizer', \
    allow_patterns=['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'config.json', 'LICENSE'])" \
    && rm -rf /app/tokenizer/.cache && mkdir -p /licenses/qwen-tokenizer && mv /app/tokenizer/LICENSE /licenses/qwen-tokenizer/

# llama-server finds its backends next to itself, and its shared libraries through LD_LIBRARY_PATH.
COPY --from=llama-build /out/llama/ /opt/llama/
COPY --from=llama-build /out/licenses/ /licenses/
COPY LICENSE /licenses/gymleader/LICENSE
COPY licenses/ /licenses/
# api.py reads only src/ and the catalog (verifier.py: ROOT / data/processed/catalog.json)
COPY src/ src/
COPY data/processed/catalog.json data/processed/catalog.json

# /models belongs to the app user, so a new named volume mounted there is writable by it.
RUN useradd --create-home --uid 1000 app && mkdir /models && chown app:app /models
USER app
ENV PATH=/opt/llama:$PATH \
    LD_LIBRARY_PATH=/opt/llama \
    TOKENIZER=/app/tokenizer \
    MODEL_DIR=/models \
    PYTHONUNBUFFERED=1
VOLUME /models
EXPOSE 8000
# Failures in the start period don't count: 30 min covers the 2.1 GB first download on a slow line.
HEALTHCHECK --start-period=30m --start-interval=5s --interval=30s --timeout=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
LABEL org.opencontainers.image.title="GymLeader" \
      org.opencontainers.image.description="Gym workouts from plain requests: a fine-tuned Qwen3.5-2B on llama.cpp b12360 (2d8a32502), checked against an exercise catalog. Web page on port 8000." \
      org.opencontainers.image.source="https://github.com/riccardoroselli/SFT-Gym-Workout-Program" \
      org.opencontainers.image.licenses="Apache-2.0"
ENTRYPOINT ["python", "/app/src/serve.py"]
