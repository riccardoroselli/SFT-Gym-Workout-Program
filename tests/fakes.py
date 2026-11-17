"""A fake llama-server and a fake tokenizer, shared by the evaluate.py and api.py tests (no model needed)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ANSWER = json.dumps({"workout": [{"name": "barbell bench press", "sets": 3, "reps": "8-12"}]})


class FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize, enable_thinking):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"


class FakeServer:
    """GET /health, GET /props, POST /completion. Completions cycle through `contents` and `stop_types`;
    requests are recorded. stop() closes the port, so later calls are refused like a server that is down."""

    def __init__(self, model_path, n_ctx=4096, stop_types=("eos",), contents=(ANSWER,)):
        self.props = {"model_path": str(model_path), "build_info": "b12360-fake",
                      "default_generation_settings": {"n_ctx": n_ctx}}
        self.stop_types, self.contents, self.requests = list(stop_types), list(contents), []
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
                i = len(server.requests) - 1
                stop = server.stop_types[i % len(server.stop_types)]
                self.reply({"content": server.contents[i % len(server.contents)], "stop_type": stop,
                            "tokens_predicted": 512 if stop == "limit" else 30,
                            "timings": {"predicted_per_second": 43.42}})

            def log_message(self, *args):  # keep pytest output quiet
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
