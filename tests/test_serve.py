"""Tests for src/serve.py without Docker, network or a model: model file + sha256 rules, tokenizer, commands,
the child environment, and the supervisor (dummy child processes)."""
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import serve

GOOD, OLD = b"the right model bytes", b"an older model"
GOOD_SHA = hashlib.sha256(GOOD).hexdigest()
ROOT = Path(serve.__file__).resolve().parent.parent


@pytest.fixture
def hub(monkeypatch):
    """A fake Hub: records downloads and writes `hub.content` where hf_hub_download would."""
    hub = SimpleNamespace(calls=[], content=GOOD, size=len(GOOD))
    monkeypatch.setattr(serve, "hf_hub_url", lambda repo, name, revision: (repo, name, revision))
    monkeypatch.setattr(serve, "get_hf_file_metadata", lambda url: SimpleNamespace(size=hub.size))

    def download(repo_id, filename, revision, local_dir):
        hub.calls.append((repo_id, filename, revision, Path(local_dir)))
        (Path(local_dir) / filename).write_bytes(hub.content)
        return str(Path(local_dir) / filename)
    monkeypatch.setattr(serve, "hf_hub_download", download)
    return hub


def cfg(tmp_path, **env):
    return serve.settings({"MODEL_DIR": str(tmp_path / "models"), "MODEL_SHA256": GOOD_SHA, **env})


def test_defaults_are_the_published_split_q8():
    c = serve.settings({})
    assert (c["MODEL_REPO"], c["MODEL_FILE"]) == ("riccardoroselli/qwen3.5-2b-gym-grpo-GGUF", "qwen3.5-2b-gym-grpo-Q8_0.gguf")
    assert c["MODEL_REVISION"] == "bbb1d3cf7f9113ec518b56c95a74702bbd83ba9b"
    assert c["MODEL_SHA256"] == "713efb40bd59dd5dc62b43f2145f540a9780ff6bfcd21006ff36dbceb7df5137"
    assert c["MODEL_DIR"] == "/models" and c["MODEL_PATH"] is None


def test_missing_file_is_downloaded_and_checked(tmp_path, hub):
    c = cfg(tmp_path)
    path = serve.model_file(c)
    assert path == tmp_path / "models" / c["MODEL_FILE"] and path.read_bytes() == GOOD
    assert hub.calls == [(c["MODEL_REPO"], c["MODEL_FILE"], c["MODEL_REVISION"], tmp_path / "models")]


def test_good_file_is_used_without_download(tmp_path, hub):
    c = cfg(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / c["MODEL_FILE"]).write_bytes(GOOD)
    assert serve.model_file(c).read_bytes() == GOOD and hub.calls == []


def test_stale_file_is_replaced(tmp_path, hub):
    c = cfg(tmp_path)
    meta = tmp_path / "models" / ".cache" / "huggingface" / "download" / f"{c['MODEL_FILE']}.metadata"
    meta.parent.mkdir(parents=True)
    meta.write_text("old record")
    (tmp_path / "models" / c["MODEL_FILE"]).write_bytes(OLD)
    assert serve.model_file(c).read_bytes() == GOOD and len(hub.calls) == 1
    assert not meta.exists()  # removed with the stale file (the fake download doesn't write one)


def test_wrong_download_is_deleted(tmp_path, hub):
    hub.content = OLD
    c = cfg(tmp_path)
    with pytest.raises(serve.StartError, match="expected MODEL_SHA256.*deleted it"):
        serve.model_file(c)
    assert not (tmp_path / "models" / c["MODEL_FILE"]).exists()


def test_model_path_is_used_as_is_and_never_deleted(tmp_path, hub):
    own = tmp_path / "mine.gguf"
    own.write_bytes(GOOD)
    assert serve.model_file(cfg(tmp_path, MODEL_PATH=str(own))) == own
    own.write_bytes(OLD)
    with pytest.raises(serve.StartError, match="set MODEL_SHA256"):
        serve.model_file(cfg(tmp_path, MODEL_PATH=str(own)))
    assert own.read_bytes() == OLD and hub.calls == []
    with pytest.raises(serve.StartError, match="not a file"):
        serve.model_file(cfg(tmp_path, MODEL_PATH=str(tmp_path / "missing.gguf")))


def test_not_enough_space(tmp_path, hub):
    hub.size = 10**15  # 1 PB
    with pytest.raises(serve.StartError, match="GB free"):
        serve.model_file(cfg(tmp_path))
    assert hub.calls == []


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="root can write anywhere")
def test_read_only_model_dir(tmp_path, hub):
    (tmp_path / "models").mkdir(mode=0o500)
    try:
        with pytest.raises(serve.StartError, match="not writable"):
            serve.model_file(cfg(tmp_path))
    finally:
        (tmp_path / "models").chmod(0o700)


def test_hub_errors_mention_hf_token(tmp_path, hub, monkeypatch):
    def private(url):
        raise RuntimeError("401 Unauthorized")
    monkeypatch.setattr(serve, "get_hf_file_metadata", private)
    with pytest.raises(serve.StartError, match="401.*HF_TOKEN"):
        serve.model_file(cfg(tmp_path))


def test_tokenizer(monkeypatch):
    calls = []
    monkeypatch.setattr(serve, "snapshot_download", lambda repo, **kw: calls.append((repo, kw)) or "/cache/qwen")
    assert serve.tokenizer_dir({"TOKENIZER": "/app/tokenizer"}) == "/app/tokenizer" and calls == []
    assert serve.tokenizer_dir({}) == "/cache/qwen"
    assert calls == [("Qwen/Qwen3.5-2B", {"revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
                                          "allow_patterns": serve.TOKENIZER_FILES, "token": False})]


def test_dockerfile_bakes_the_same_tokenizer():
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text()
    assert serve.TOKENIZER_REVISION in dockerfile and all(f in dockerfile for f in serve.TOKENIZER_FILES)


def test_commands_and_child_env():
    model = Path("/models/qwen3.5-2b-gym-grpo-Q8_0.gguf")
    c = serve.settings({"LLAMA_ARGS": "-t 6 --log-colors off"})
    assert serve.llama_command(model, c) == ["llama-server", "-m", str(model), "--host", "127.0.0.1", "--port", "8080",
                                             "-c", "4096", "-np", "1", "--no-webui", "-t", "6", "--log-colors", "off"]
    assert serve.api_command(c)[-4:] == ["--host", "0.0.0.0", "--port", "8000"]
    env = serve.child_env({"HF_TOKEN": "secret", "MAX_RETRIES": "2"}, model, "/app/tokenizer")
    assert "HF_TOKEN" not in env and env["MAX_RETRIES"] == "2" and env["HF_HUB_OFFLINE"] == "1"
    assert env["LLAMA_SERVER_URL"] == "http://127.0.0.1:8080" and env["TOKENIZER"] == "/app/tokenizer"
    assert env["EXPECTED_MODEL_FILE"] == model.name
    assert serve.child_env({"EXPECTED_MODEL_FILE": "x.gguf"}, model, "t")["EXPECTED_MODEL_FILE"] == "x.gguf"


def child(code):
    return subprocess.Popen([sys.executable, "-c", code])


def test_a_dying_child_stops_the_other():
    procs = {"llama-server": child("import time; time.sleep(60)"), "api": child("import sys; sys.exit(3)")}
    try:
        with pytest.raises(serve.StartError, match="api stopped with exit code 3"):
            serve.supervise(procs, poll=0.05)
    finally:
        serve.stop_all(procs, grace=5)
    assert procs["llama-server"].returncode is not None  # terminated, not left running


def test_wait_until_up_fails_if_the_process_exits():
    with pytest.raises(serve.StartError, match="llama-server exited with code 2"):
        serve.wait_until_up("llama-server", child("import sys; sys.exit(2)"), "http://127.0.0.1:9/health", poll=0.05)


def test_missing_binary():
    with pytest.raises(serve.StartError, match="not found on PATH"):
        serve.start("llama-server", ["gymleader-no-such-binary"], {})


def run_local(tmp_path, *, fake_llama, **env):
    """scripts/run_local.sh with PATH = this python + (optionally) a fake llama-server; returns the finished process."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    if fake_llama:
        (bin_dir / "llama-server").write_text("#!/bin/sh\necho 'version: 1.0.0 (build 9999, commit abc)'\n")
        (bin_dir / "llama-server").chmod(0o755)
    path = f"{bin_dir}:{Path(sys.executable).parent}:/usr/bin:/bin"
    return subprocess.run(["bash", str(ROOT / "scripts" / "run_local.sh")], capture_output=True, text=True,
                          env={"PATH": path, "HOME": os.environ.get("HOME", str(tmp_path)), **env}, timeout=120)


@pytest.mark.skipif(os.name != "posix", reason="bash script")
def test_run_local_needs_llama_server(tmp_path):
    r = run_local(tmp_path, fake_llama=False)
    assert r.returncode == 1 and "llama-server is not on PATH" in r.stderr


@pytest.mark.skipif(os.name != "posix", reason="bash script")
def test_run_local_warns_on_other_build_then_runs_serve(tmp_path):
    own = tmp_path / "mine.gguf"
    own.write_bytes(OLD)
    r = run_local(tmp_path, fake_llama=True, MODEL_PATH=str(own))
    assert "WARNING: not llama.cpp build 12360" in r.stderr
    assert r.returncode == 1 and "set MODEL_SHA256" in r.stdout  # serve.py ran, checked the file and stopped
    assert own.read_bytes() == OLD
