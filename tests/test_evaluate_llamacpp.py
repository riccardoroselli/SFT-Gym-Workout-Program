"""Tests for the llamacpp backend of src/evaluate.py generate, against a fake llama-server (no model needed)."""
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import evaluate
from evaluate import TEST, build_parser, build_prompt, few_shot_turns, generate, read_jsonl, score

MLX_KEYS = ["meta", "request", "output", "truncated", "tokens", "tps", "model", "adapter", "few_shot"]
ANSWER = json.dumps({"workout": [{"name": "barbell bench press", "sets": 3, "reps": "8-12"}]})


class FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize, enable_thinking):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"


class FakeServer:
    """GET /health, GET /props, POST /completion. Completions cycle through `stop_types`; requests are recorded."""

    def __init__(self, model_path, n_ctx=4096, stop_types=("eos",)):
        self.props = {"model_path": str(model_path), "build_info": "b12360-fake",
                      "default_generation_settings": {"n_ctx": n_ctx}}
        self.stop_types, self.requests = list(stop_types), []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def reply(self, obj):
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.reply({"status": "ok"} if self.path == "/health" else server.props)

            def do_POST(self):
                server.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                stop = server.stop_types[(len(server.requests) - 1) % len(server.stop_types)]
                self.reply({"content": ANSWER, "stop_type": stop, "tokens_predicted": 512 if stop == "limit" else 30,
                            "timings": {"predicted_per_second": 43.42}})

            def log_message(self, *args):  # keep pytest output quiet
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """-> (gguf path, start(**FakeServer kwargs) -> server). Outputs go to tmp_path; servers are shut down after."""
    monkeypatch.setattr(evaluate, "OUTPUTS", tmp_path)
    gguf = tmp_path / "model-Q8_0.gguf"
    gguf.write_bytes(b"")
    servers = []

    def start(**kwargs):
        servers.append(FakeServer(kwargs.pop("model_path", gguf), **kwargs))
        return servers[-1]
    yield gguf, start
    for s in servers:
        s.httpd.shutdown()


def run(gguf, server, limit=2, k=0):
    generate("fake", str(gguf), None, k, limit=limit, backend="llamacpp", server_url=server.url,
             tokenizer=FakeTokenizer())
    return read_jsonl(evaluate.OUTPUTS / "fake.jsonl")


def test_row_format_request_and_score(setup):
    gguf, start = setup
    server = start()
    rows = run(gguf, server, k=1)
    assert list(rows[0]) == MLX_KEYS + ["backend", "model_file", "server_build"]
    assert rows[0]["backend"] == "llamacpp" and rows[0]["model_file"] == "model-Q8_0.gguf"
    assert rows[0]["server_build"] == "b12360-fake" and rows[0]["model"] == str(gguf) and rows[0]["adapter"] is None
    assert rows[0]["tokens"] == 30 and rows[0]["tps"] == 43.4 and rows[0]["few_shot"] == 1

    # the raw build_prompt() text with the few-shot turns, greedy, no penalties, no prompt cache
    ex = read_jsonl(TEST)[0]
    expected = build_prompt(FakeTokenizer(), [ex["messages"][0], *few_shot_turns(1), ex["messages"][1]])
    req = server.requests[0]
    assert req["prompt"] == expected
    assert req["n_predict"] == evaluate.MAX_NEW_TOKENS == 512 and req["stop"] == ["<|im_end|>"]
    assert req["cache_prompt"] is False and req["temperature"] == 0
    assert (req["repeat_penalty"], req["presence_penalty"], req["frequency_penalty"], req["dry_multiplier"]) == (1, 0, 0, 0)

    s = score("fake", outputs_dir=evaluate.OUTPUTS, results_dir=evaluate.OUTPUTS)  # score works unchanged
    assert s["n"] == 2 and s["truncated"] == 0.0


def test_truncated_from_stop_type(setup):
    gguf, start = setup
    rows = run(gguf, start(stop_types=("eos", "limit", "word")), limit=3)
    assert [r["truncated"] for r in rows] == [False, True, False]
    assert [r["tokens"] for r in rows] == [30, 512, 30]


def test_unknown_stop_type_stops_the_run(setup):
    gguf, start = setup
    with pytest.raises(SystemExit, match="stop_type 'none'"):
        run(gguf, start(stop_types=("none",)))


def test_props_model_mismatch(setup, tmp_path):
    gguf, start = setup
    other = tmp_path / "model-Q4_K_M.gguf"
    other.write_bytes(b"")
    with pytest.raises(SystemExit, match="has loaded .*model-Q4_K_M.gguf.*not --model"):
        run(gguf, start(model_path=other))


def test_small_server_context_stops(setup):
    gguf, start = setup
    with pytest.raises(SystemExit, match="context per slot is 1024, need at least 2048"):
        run(gguf, start(n_ctx=1024))


def test_unreachable_server_fails_fast(setup):
    gguf, _ = setup
    with socket.socket() as s:  # a free port with nothing listening
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(SystemExit, match="llama-server not ready"):
        generate("fake", str(gguf), None, 0, limit=1, backend="llamacpp", server_url=f"http://127.0.0.1:{port}",
                 tokenizer=FakeTokenizer())


def test_mlx_is_the_default():
    args = build_parser().parse_args(["generate", "--run", "x"])
    assert args.backend == "mlx" and args.model == evaluate.DEFAULT_MODEL
    assert args.server_url == "http://127.0.0.1:8080" and args.tokenizer == "riccardoroselli/qwen3.5-2b-gym-grpo"
