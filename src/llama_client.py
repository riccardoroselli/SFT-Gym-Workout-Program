"""Client for a running llama-server (llama.cpp), shared by evaluate.py and api.py. Standard library only.

    props = check_server("http://127.0.0.1:8080")                       # /health + /props, context per slot >= 2048
    text, truncated, tokens, tps = complete(url, prompt)                # greedy, exactly as evaluate.py
    complete(url, prompt, temperature=0.7, seed=1)                      # overrides go on top of the greedy fields

Every problem raises LlamaServerError: callers decide whether that stops a script or becomes an HTTP 503.
"""
import json
import urllib.request

MAX_NEW_TOKENS = 512
MIN_SERVER_CTX = 2048  # per slot: prompt (with few-shot turns) + MAX_NEW_TOKENS must fit
# Greedy, and every repetition penalty off whatever flags the server was started with, so the answers match mlx's.
GREEDY = {"temperature": 0, "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
          "dry_multiplier": 0.0}
STOP_TYPES = {"eos": False, "word": False, "limit": True}  # llama-server stop_type -> truncated


class LlamaServerError(Exception):
    """llama-server is down, still loading, too small, or answered something unexpected."""


def http_json(url, body=None, timeout=600):
    """GET (body None) or POST a JSON body -> the parsed JSON answer."""
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def check_server(url):
    """-> the server's /props. Raises if /health fails (503 while the model loads) or the context per slot is too small."""
    url = url.rstrip("/")
    try:
        http_json(f"{url}/health", timeout=5)
        props = http_json(f"{url}/props", timeout=5)
    except (OSError, ValueError) as e:  # URLError, HTTPError, timeouts and refused connections are all OSError
        raise LlamaServerError(f"llama-server not ready at {url}: {e}") from e
    n_ctx = props["default_generation_settings"]["n_ctx"]  # per slot: -np N splits -c by N
    if n_ctx < MIN_SERVER_CTX:
        raise LlamaServerError(f"llama-server context per slot is {n_ctx}, need at least {MIN_SERVER_CTX}: "
                               f"restart it with -c 4096 -np 1")
    return props


def complete(url, prompt, **overrides):
    """Raw prompt to /completion (no chat template on the server side) -> (text, truncated, tokens, tokens/s)."""
    body = {"prompt": prompt, "n_predict": MAX_NEW_TOKENS, "stop": ["<|im_end|>"], "cache_prompt": False,
            **GREEDY, **overrides}
    try:
        out = http_json(f"{url.rstrip('/')}/completion", body)
    except (OSError, ValueError) as e:
        raise LlamaServerError(f"llama-server request failed at {url}: {e}") from e
    if out.get("stop_type") not in STOP_TYPES:
        raise LlamaServerError(f"llama-server returned stop_type {out.get('stop_type')!r}, "
                               f"expected one of {list(STOP_TYPES)}")
    # tokens_predicted counts the <|im_end|> token too, like mlx's generation_tokens
    return (out["content"], STOP_TYPES[out["stop_type"]], out["tokens_predicted"],
            round(out["timings"]["predicted_per_second"], 1))
