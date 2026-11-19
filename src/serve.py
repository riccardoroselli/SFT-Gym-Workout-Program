"""Starts the whole GymLeader system in one process tree: model file -> llama-server -> API + web page.

    python src/serve.py        # the Docker image's entrypoint; also works on a Mac with llama-server on PATH

1. The model: MODEL_PATH if set (your own file, never deleted), else MODEL_DIR/MODEL_FILE, downloaded from
   MODEL_REPO at MODEL_REVISION if missing. Its sha256 must equal MODEL_SHA256 before anything starts.
   A wrong file in MODEL_DIR (an older version) is deleted and downloaded again; a wrong download is deleted.
2. The tokenizer: TOKENIZER if set (the image sets it), else the 4 files from Qwen/Qwen3.5-2B at a pinned revision.
3. llama-server on 127.0.0.1:8080 (not reachable from outside), flags as in every eval (-c 4096 -np 1) + LLAMA_ARGS.
4. uvicorn api:app on HOST:PORT (0.0.0.0:8000) with HF_HUB_OFFLINE=1, after llama-server's /health answers.
If either process stops, the other is stopped and serve.py exits non-zero. SIGTERM/Ctrl-C stop both, exit 0.

Environment: MODEL_PATH, MODEL_DIR, MODEL_REPO, MODEL_FILE, MODEL_REVISION, MODEL_SHA256, HF_TOKEN (only while
the model repo is private), LLAMA_ARGS (e.g. "-t 6", or "-ngl all" on a Mac), HOST, PORT, and api.py's own
MAX_RETRIES, RETRY_TEMPERATURE, TOKENIZER, EXPECTED_MODEL_FILE (default: the model's file name).
"""
import hashlib
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url, snapshot_download
from huggingface_hub.utils import disable_progress_bars

SRC = Path(__file__).resolve().parent
LLAMA_URL = "http://127.0.0.1:8080"
DEFAULTS = {
    "MODEL_DIR": "/models",
    "MODEL_REPO": "riccardoroselli/qwen3.5-2b-gym-grpo-GGUF",
    "MODEL_FILE": "qwen3.5-2b-gym-grpo-Q8_0.gguf",
    "MODEL_REVISION": "bbb1d3cf7f9113ec518b56c95a74702bbd83ba9b",  # the commit with the split, chat-ready files (17.1b)
    "MODEL_SHA256": "713efb40bd59dd5dc62b43f2145f540a9780ff6bfcd21006ff36dbceb7df5137",
    "HOST": "0.0.0.0",
    "PORT": "8000",
    "LLAMA_ARGS": "",
}
# The same 4 files docker/Dockerfile bakes into the image (16c checked they give the same prompts and token IDs).
TOKENIZER_REPO, TOKENIZER_REVISION = "Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json"]


class StartError(Exception):
    """Something the user must fix; main() prints it and exits 1."""


class Stop(Exception):
    """SIGTERM (docker stop)."""


def log(message):
    print(f"gymleader: {message}", flush=True)


def settings(env):
    return {**{k: env.get(k) or v for k, v in DEFAULTS.items()}, "MODEL_PATH": env.get("MODEL_PATH")}


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def remove(path):
    """The file and huggingface_hub's local_dir record of it, so the next download starts clean."""
    path.unlink(missing_ok=True)
    (path.parent / ".cache" / "huggingface" / "download" / f"{path.name}.metadata").unlink(missing_ok=True)


def check(path, expected, delete):
    log(f"checking the sha256 of {path.name}")
    actual = sha256(path)
    if actual != expected.lower():
        if delete:
            remove(path)
        hint = "deleted it; start again to retry the download" if delete else \
            "if this is the file you want, set MODEL_SHA256 to its sha256"
        raise StartError(f"{path} has sha256 {actual}, expected MODEL_SHA256 {expected}: {hint}")
    log("checksum OK")


def download(cfg, path):
    folder, repo, name, rev = path.parent, cfg["MODEL_REPO"], cfg["MODEL_FILE"], cfg["MODEL_REVISION"]
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise StartError(f"can't create MODEL_DIR {folder}: {e}") from e
    if not os.access(folder, os.W_OK):
        raise StartError(f"MODEL_DIR {folder} is not writable by uid {os.getuid()}: mount a volume there "
                         f"(-v gymleader-models:{folder}) or give that user write access")
    try:  # token=None: huggingface_hub reads HF_TOKEN from the environment itself
        size = get_hf_file_metadata(hf_hub_url(repo, name, revision=rev)).size
    except Exception as e:
        raise StartError(f"can't find {name} in {repo} at revision {rev}: {e}. "
                         "If the repo is private, set HF_TOKEN.") from e
    free = shutil.disk_usage(folder).free
    if size and free < size:
        raise StartError(f"{folder} has {free / 1e9:.2f} GB free, the model needs {size / 1e9:.2f} GB")
    log(f"downloading {name} ({size / 1e9:.2f} GB) from {repo} at {rev[:8]} into {folder}")
    try:
        hf_hub_download(repo_id=repo, filename=name, revision=rev, local_dir=folder)
    except Exception as e:  # a partial download stays in folder/.cache, and the next start resumes it
        raise StartError(f"download of {name} failed: {e}") from e


def model_file(cfg):
    """-> the checked model file. Raises StartError."""
    if cfg["MODEL_PATH"]:
        path = Path(cfg["MODEL_PATH"])
        if not path.is_file():
            raise StartError(f"MODEL_PATH {path} is not a file (is its folder mounted?)")
        check(path, cfg["MODEL_SHA256"], delete=False)  # the user's own file: never deleted
        return path
    path = Path(cfg["MODEL_DIR"]) / cfg["MODEL_FILE"]
    if path.is_file():
        if sha256(path) == cfg["MODEL_SHA256"].lower():
            log(f"found {path}, checksum OK")
            return path
        log(f"{path} has a different sha256 (an older version?): deleting it and downloading again")
        remove(path)
    download(cfg, path)
    check(path, cfg["MODEL_SHA256"], delete=True)
    return path


def tokenizer_dir(env):
    if env.get("TOKENIZER"):
        return env["TOKENIZER"]
    log(f"getting the tokenizer files from {TOKENIZER_REPO} at {TOKENIZER_REVISION[:8]}")
    # 4 small files: their bars only print "Download complete 0.00B / Reconstruction complete" lines at exit. Off here
    # only (huggingface_hub docs, Utilities > Configure progress bars); the 2 GB model download keeps its bar.
    with disable_progress_bars():
        return snapshot_download(TOKENIZER_REPO, revision=TOKENIZER_REVISION, allow_patterns=TOKENIZER_FILES,
                                 token=False)


def child_env(env, model, tokenizer):
    child = {k: v for k, v in env.items() if k != "HF_TOKEN"}  # nothing after this point downloads
    child.update(HF_HUB_OFFLINE="1", LLAMA_SERVER_URL=LLAMA_URL, TOKENIZER=tokenizer,
                 EXPECTED_MODEL_FILE=env.get("EXPECTED_MODEL_FILE") or model.name)
    return child


def llama_command(model, cfg):
    return ["llama-server", "-m", str(model), "--host", "127.0.0.1", "--port", LLAMA_URL.rsplit(":", 1)[1],
            "-c", "4096", "-np", "1", "--no-webui", *shlex.split(cfg["LLAMA_ARGS"])]


def api_command(cfg):
    return [sys.executable, "-m", "uvicorn", "api:app", "--app-dir", str(SRC), "--host", cfg["HOST"], "--port", cfg["PORT"]]


def start(name, command, env):
    try:  # own session: Ctrl-C reaches serve.py only, which then stops the children in order
        return subprocess.Popen(command, env=env, start_new_session=True)
    except FileNotFoundError as e:
        raise StartError(f"{name}: {command[0]} not found on PATH") from e


def wait_until_up(name, proc, url, poll=0.5):
    """Until url answers 200; raises StartError if the process exits first. No time limit: a big model on a slow
    disk can take minutes, and Docker's HEALTHCHECK start period is the limit."""
    while proc.poll() is None:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except OSError:  # not listening yet, or 503 while loading (HTTPError is an OSError)
            pass
        time.sleep(poll)
    raise StartError(f"{name} exited with code {proc.returncode} while starting (see its log above)")


def supervise(procs, poll=1.0):
    while True:
        for name, proc in procs.items():
            if proc.poll() is not None:
                raise StartError(f"{name} stopped with exit code {proc.returncode}")
        time.sleep(poll)


def stop_all(procs, grace=8):
    for proc in reversed(list(procs.values())):  # the API first, then llama-server
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(grace)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def _on_sigterm(signum, frame):
    raise Stop


def main():
    signal.signal(signal.SIGTERM, _on_sigterm)
    procs = {}
    try:
        cfg = settings(os.environ)
        model = model_file(cfg)
        env = child_env(os.environ, model, tokenizer_dir(os.environ))
        log(f"loading the model {model.name} in llama-server")
        procs["llama-server"] = start("llama-server", llama_command(model, cfg), env)
        wait_until_up("llama-server", procs["llama-server"], f"{LLAMA_URL}/health")
        log("model loaded, starting the API")
        procs["api"] = start("api", api_command(cfg), env)
        wait_until_up("api", procs["api"], f"http://127.0.0.1:{cfg['PORT']}/health")
        log(f"ready at http://127.0.0.1:{cfg['PORT']}")
        supervise(procs)
    except (Stop, KeyboardInterrupt):
        log("stopping")
        return 0
    except StartError as e:
        log(f"ERROR: {e}")
        return 1
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # a second signal must not cut the clean stop short
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        stop_all(procs)


if __name__ == "__main__":
    sys.exit(main())
