#!/usr/bin/env bash
# GymLeader on a Mac without Docker: llama-server on Metal + the API + the web page at http://127.0.0.1:8000
#   conda activate GymLeaderProject && scripts/run_local.sh          (Ctrl-C stops both processes)
# src/serve.py does the rest: it uses models/gguf/qwen3.5-2b-gym-grpo-Q8_0.gguf if its sha256 matches, otherwise
# downloads it there (a different file with that name is deleted first; set MODEL_PATH to use your own file as is),
# and gets the tokenizer files from Qwen/Qwen3.5-2B unless TOKENIZER is set. Its other variables pass through.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "run_local: python is $(command -v python)"
python - <<'EOF' || { echo "run_local: activate an env with requirements-serve.txt installed" >&2; exit 1; }
import importlib, re
names = [re.split(r"[=<>!~ ;\[]", l.strip())[0] for l in open("requirements-serve.txt") if l.strip() and l[0] != "#"]
missing = []
for name in names:
    try:
        importlib.import_module(name.lower().replace("-", "_"))
    except ImportError:
        missing.append(name)
if missing:
    raise SystemExit(f"run_local: missing python packages: {', '.join(missing)}")
EOF

if ! command -v llama-server >/dev/null; then
    echo "run_local: llama-server is not on PATH (build llama.cpp tag b12360 and add its bin/ to PATH)" >&2
    exit 1
fi
version=$(llama-server --version 2>&1 || true)
echo "run_local: $(command -v llama-server): $(head -n1 <<<"$version")"
if ! grep -qF "build 12360," <<<"$version"; then
    echo "run_local: WARNING: not llama.cpp build 12360, the build every eval used; answers may differ" >&2
fi

# 127.0.0.1, not serve.py's 0.0.0.0 (right in a container, but here it would open the API to the whole network)
export MODEL_DIR="${MODEL_DIR:-models/gguf}" LLAMA_ARGS="${LLAMA_ARGS:--ngl all}" HOST="${HOST:-127.0.0.1}"
exec python src/serve.py  # exec: serve.py gets Ctrl-C itself and stops llama-server and the API
