"""GymLeader API: a workout request in plain words -> a workout (or an error) that the verifier has checked.

    llama-server -m "$PWD/models/gguf/qwen3.5-2b-gym-grpo-Q8_0.gguf" -ngl all -c 4096 -np 1 --port 8080
    HF_HUB_OFFLINE=1 uvicorn api:app --app-dir src --port 8000
    curl -s localhost:8000/workout -H 'Content-Type: application/json' \
         -d '{"request": "chest workout, medium difficulty, free weights"}'

POST /workout:
1. parse the text into body part, equipment group, difficulty, count (request_parser.py); unreadable -> 422.
   Everyday words like "biceps" or "dumbbells" are serving aliases: each one is reported in "interpreted_as";
2. impossible request (is_feasible() false) -> data-v1's error answer, without calling the model ("guard");
3. otherwise the model, greedy, exactly as evaluate.py's llamacpp backend. It gets the user's text unchanged if it
   uses only spec words, else a canonical sentence in spec words ("model_input"). If verify() says it isn't all correct,
   up to MAX_RETRIES samples at RETRY_TEMPERATURE with seed k. First all-correct answer wins, else the best reward
   (earliest on a tie) with verified=false. llama-server down -> 503.

Settings, from environment variables: LLAMA_SERVER_URL (http://127.0.0.1:8080), MAX_RETRIES (3),
RETRY_TEMPERATURE (0.7), TOKENIZER (riccardoroselli/qwen3.5-2b-gym-grpo, from the local HF cache),
EXPECTED_MODEL_FILE (optional: the .gguf file name llama-server must have loaded).

GET / serves the web page in src/web/ (plain HTML/CSS/JS under /static). The page makes no outside requests;
/docs (FastAPI's Swagger UI) loads its script and CSS from a CDN.
"""
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from analyze_errors import first_failed
from evaluate import DEFAULT_SERVER, DEFAULT_TOKENIZER, build_prompt
from llama_client import LlamaServerError, check_server, complete
from make_dataset import SYSTEM, error_answer
from request_parser import EXAMPLE_REQUEST, RequestError, model_input, parse_request
from verifier import _parse, is_feasible, verify

MAX_REQUEST_CHARS = 500  # keeps the prompt far below the server's 2048-token context per slot
# Retries sample like the GRPO probe and training: temperature only, llama-server's default filters off.
NO_FILTERS = {"top_k": 0, "top_p": 1.0, "min_p": 0.0}
WEB = Path(__file__).resolve().parent / "web"


def load_settings():
    env = os.environ.get
    settings = {"server_url": env("LLAMA_SERVER_URL", DEFAULT_SERVER).rstrip("/"),
                "max_retries": int(env("MAX_RETRIES", "3")),
                "retry_temperature": float(env("RETRY_TEMPERATURE", "0.7")),
                "tokenizer": env("TOKENIZER", DEFAULT_TOKENIZER),
                "expected_model_file": env("EXPECTED_MODEL_FILE") or None}
    if settings["max_retries"] < 0 or settings["retry_temperature"] <= 0:
        raise ValueError(f"MAX_RETRIES must be >= 0 and RETRY_TEMPERATURE > 0, got {settings}")
    return settings


def load_tokenizer(name):
    from transformers import AutoTokenizer  # the prompt must be exactly build_prompt(), as in evaluate.py
    return AutoTokenizer.from_pretrained(name, local_files_only=True)


@asynccontextmanager
async def lifespan(app):
    """Startup checks, like evaluate.py: any failure stops the app before it takes requests."""
    settings = load_settings()
    props = check_server(settings["server_url"])
    model_file = Path(props.get("model_path", "")).name  # the name only: the server may run in another container
    if settings["expected_model_file"] and model_file != settings["expected_model_file"]:
        raise RuntimeError(f"llama-server at {settings['server_url']} has loaded {model_file!r}, "
                           f"not EXPECTED_MODEL_FILE {settings['expected_model_file']!r}")
    app.state.settings, app.state.tokenizer = settings, load_tokenizer(settings["tokenizer"])
    app.state.model_file, app.state.server_build = model_file, props.get("build_info")
    yield


app = FastAPI(title="GymLeader", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB), name="static")  # not in /docs


@app.get("/", include_in_schema=False)
def page():
    return FileResponse(WEB / "index.html")


class WorkoutRequest(BaseModel):
    request: str = Field(min_length=1, max_length=MAX_REQUEST_CHARS)


# OpenAPI-level example (not JSON Schema "examples"): the one Swagger UI prefills in "Try it out"
REQUEST_EXAMPLES = {"valid": {"summary": "A valid request", "value": {"request": EXAMPLE_REQUEST}}}


class Attempt(BaseModel):
    """One model call: greedy first, then the retries."""
    source: str  # model_greedy | model_retry_<k>
    temperature: float
    seed: int | None  # None: greedy sends no seed
    raw_output: str
    tokens: int
    tps: float
    truncated: bool  # stopped by the 512-token limit, not by the model
    ms: int
    reward: float
    all_correct: bool
    first_failed: str  # analyze_errors.py's order: parse, names, body part, ..., or "passed"


class WorkoutResponse(BaseModel):
    request: dict
    workout: list | None  # not list[dict]: an odd model answer must come back as it is, not as a 500
    error: str | None
    raw_output: str
    interpreted_as: list[str]  # one note per serving alias used, e.g. "dumbbells → free weights (...)"
    model_input: str | None  # the user text sent to the model; None when the guard answered
    source: str  # guard | model_greedy | model_retry_<k>
    verified: bool
    checks: dict
    model_calls: int
    attempts: list[Attempt]  # every model call, in order; empty when the guard answered
    timings_ms: dict
    model_file: str
    server_build: str | None


class HealthResponse(BaseModel):
    status: str
    model_file: str
    server_build: str | None


class Unavailable(BaseModel):
    status: str
    detail: str


def ask_model(text, req):
    """-> (attempts, index of the chosen one, its verify result). Raises HTTP 503 if llama-server fails."""
    settings = app.state.settings
    prompt = build_prompt(app.state.tokenizer, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}])
    attempts, results = [], []
    for k in range(settings["max_retries"] + 1):
        temperature = settings["retry_temperature"] if k else 0.0
        overrides = {"temperature": temperature, "seed": k, **NO_FILTERS} if k else {}
        start = time.perf_counter()
        try:
            output, truncated, tokens, tps = complete(settings["server_url"], prompt, **overrides)
        except LlamaServerError as e:
            raise HTTPException(503, detail=str(e)) from e
        ms = round((time.perf_counter() - start) * 1000)
        result = verify(req, output)
        results.append(result)
        attempts.append({"source": f"model_retry_{k}" if k else "model_greedy", "temperature": temperature,
                         "seed": k or None, "raw_output": output, "tokens": tokens, "tps": tps, "truncated": truncated,
                         "ms": ms, "reward": result["reward"], "all_correct": bool(result["all_correct"]),
                         "first_failed": first_failed({**req, "feasible": True}, result)})
        if result["all_correct"]:
            break
    # all correct: the last attempt; otherwise the best reward, and max() keeps the earliest of equal rewards
    best = len(attempts) - 1 if results[-1]["all_correct"] else max(range(len(attempts)), key=lambda i: results[i]["reward"])
    return attempts, best, results[best]


@app.post("/workout", response_model=WorkoutResponse)
def workout(body: Annotated[WorkoutRequest, Body(openapi_examples=REQUEST_EXAMPLES)]):
    start = time.perf_counter()
    try:
        req, notes = parse_request(body.request)
    except RequestError as e:
        raise HTTPException(422, detail=e.detail()) from e
    feasible = is_feasible(req)
    if feasible:
        text = model_input(body.request, req, notes)
        attempts, best, result = ask_model(text, req)
        source, output = attempts[best]["source"], attempts[best]["raw_output"]
    else:  # the catalog decides, not the model: the error answer is exact and costs no model call
        output, text, attempts = error_answer(req), None, []
        source, result = "guard", verify(req, output)
    obj = _parse(output)[0] if result["parse_ok"] else {}
    return {"request": {**req, "feasible": feasible}, "workout": obj.get("workout"), "error": obj.get("error"),
            "raw_output": output, "interpreted_as": notes, "model_input": text, "source": source,
            "verified": bool(result["all_correct"]), "checks": result, "model_calls": len(attempts), "attempts": attempts,
            "timings_ms": {"model": sum(a["ms"] for a in attempts),
                           "total": round((time.perf_counter() - start) * 1000)},
            "model_file": app.state.model_file, "server_build": app.state.server_build}


@app.get("/health", response_model=HealthResponse, responses={503: {"model": Unavailable}})
def health():
    """200 if llama-server answers its own /health (and still has enough context), else 503."""
    try:
        check_server(app.state.settings["server_url"])
    except LlamaServerError as e:
        return JSONResponse(status_code=503, content={"status": "unavailable", "detail": str(e)})
    return {"status": "ok", "model_file": app.state.model_file, "server_build": app.state.server_build}
