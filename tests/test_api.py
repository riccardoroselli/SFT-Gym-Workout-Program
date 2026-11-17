"""Tests for src/api.py against a fake llama-server: guard, greedy, retries, best-of, 422, 503, /health, startup."""
import contextlib
import json

import pytest
from fastapi.testclient import TestClient

import api
from evaluate import build_prompt
from fakes import FakeServer, FakeTokenizer
from llama_client import GREEDY, LlamaServerError
from make_dataset import SYSTEM, error_answer
from request_parser import EXAMPLE_REQUEST

REQUEST = "Chest workout, medium difficulty, free weights"  # no count: 4
NAMES = ["barbell bench press", "dumbbell bench press", "barbell incline bench press", "barbell decline bench press"]


def answer(names):
    return json.dumps({"workout": [{"name": n, "sets": 3, "reps": "8-12"} for n in names]})


GOOD = answer(NAMES)                               # reward 1.0
DUP = answer(NAMES[:3] + NAMES[:1])                # a repeat: reward 0.91
INVENTED = answer(NAMES[:3] + ["dumbbell moon press"])  # 3 of 4 names exist: reward 0.865
BROKEN = "not json"                                # reward 0.0


def prompt(user_text):
    return build_prompt(FakeTokenizer(), [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text}])


GREEDY_BODY = {"prompt": prompt(REQUEST), "n_predict": 512, "stop": ["<|im_end|>"], "cache_prompt": False, **GREEDY}


def retry_body(k):
    return {**GREEDY_BODY, "temperature": 0.7, "seed": k, "top_k": 0, "top_p": 1.0, "min_p": 0.0}


@pytest.fixture
def start(monkeypatch, tmp_path):
    """-> start(contents, n_ctx, **env) -> (client, server). The app's startup (lifespan) runs inside TestClient."""
    for name in ["LLAMA_SERVER_URL", "MAX_RETRIES", "RETRY_TEMPERATURE", "TOKENIZER", "EXPECTED_MODEL_FILE"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(api, "load_tokenizer", lambda name: FakeTokenizer())
    with contextlib.ExitStack() as stack:
        def _start(contents=(GOOD,), n_ctx=4096, **env):
            server = FakeServer(tmp_path / "model-Q8_0.gguf", n_ctx=n_ctx, contents=contents)
            stack.callback(server.stop)
            monkeypatch.setenv("LLAMA_SERVER_URL", server.url)
            for name, value in env.items():
                monkeypatch.setenv(name, value)
            return stack.enter_context(TestClient(api.app)), server
        yield _start


def post(client, text=REQUEST):
    return client.post("/workout", json={"request": text})


def test_guard_answers_impossible_requests_without_the_model(start):
    client, server = start()
    r = post(client, "Forearms workout, easy difficulty, bands, 3 exercises")  # only 2 forearms band exercises
    assert r.status_code == 200
    body = r.json()
    meta = {"body_part": "forearms", "equipment_group": "bands", "difficulty": "easy", "n_exercises": 3}
    assert body["request"] == {**meta, "feasible": False}
    assert body["raw_output"] == error_answer(meta) and body["error"] == json.loads(error_answer(meta))["error"]
    assert body["workout"] is None and body["source"] == "guard" and body["verified"] is True
    assert body["model_calls"] == 0 and body["timings_ms"]["model"] == 0
    assert body["model_input"] is None and body["interpreted_as"] == [] and body["attempts"] == []
    assert server.requests == []


def test_greedy_pass(start):
    client, server = start(contents=(GOOD,))
    body = post(client).json()
    assert body["source"] == "model_greedy" and body["verified"] is True and body["model_calls"] == 1
    assert body["workout"] == json.loads(GOOD)["workout"] and body["error"] is None
    assert body["checks"]["reward"] == 1.0 and body["checks"]["all_correct"] is True
    assert body["request"] == {"body_part": "chest", "equipment_group": "free weights", "difficulty": "medium",
                               "n_exercises": 4, "feasible": True}
    assert body["model_file"] == "model-Q8_0.gguf" and body["server_build"] == "b12360-fake"
    assert body["interpreted_as"] == [] and body["model_input"] == REQUEST  # spec words: the text goes unchanged
    assert server.requests == [GREEDY_BODY]  # exactly evaluate.py's llamacpp request


def test_aliases_send_the_canonical_sentence_and_say_so(start):
    client, server = start(MAX_RETRIES="0")
    body = post(client, "biceps workout, easy, dumbbells, 5 exercises").json()
    canonical = "Arms workout, easy difficulty, free weights, 5 exercises"
    assert body["request"] == {"body_part": "arms", "equipment_group": "free weights", "difficulty": "easy",
                               "n_exercises": 5, "feasible": True}
    assert body["model_input"] == canonical and [n.split(" → ")[0] for n in body["interpreted_as"]] == ["biceps", "dumbbells"]
    assert server.requests == [{**GREEDY_BODY, "prompt": prompt(canonical)}]


def test_retry_pass_at_k2(start):
    client, server = start(contents=(DUP, DUP, GOOD))
    body = post(client).json()
    assert body["source"] == "model_retry_2" and body["verified"] is True and body["model_calls"] == 3
    assert body["raw_output"] == GOOD
    assert server.requests == [GREEDY_BODY, retry_body(1), retry_body(2)]
    a = body["attempts"]
    assert [(x["source"], x["temperature"], x["seed"]) for x in a] == [
        ("model_greedy", 0.0, None), ("model_retry_1", 0.7, 1), ("model_retry_2", 0.7, 2)]
    assert [x["raw_output"] for x in a] == [DUP, DUP, GOOD]
    assert [(x["all_correct"], x["first_failed"]) for x in a] == [(False, "duplicates"), (False, "duplicates"), (True, "passed")]
    assert [x["reward"] for x in a] == [pytest.approx(0.91), pytest.approx(0.91), 1.0]
    assert all(x["tokens"] == 30 and x["tps"] == 43.4 and x["truncated"] is False and x["ms"] >= 0 for x in a)
    assert body["timings_ms"]["model"] == sum(x["ms"] for x in a)


def test_all_fail_returns_the_best_reward(start):
    client, server = start(contents=(INVENTED, BROKEN, DUP, BROKEN))
    body = post(client).json()
    assert body["source"] == "model_retry_2" and body["verified"] is False and body["model_calls"] == 4
    assert body["raw_output"] == DUP and body["checks"]["reward"] == pytest.approx(0.91)
    assert server.requests == [GREEDY_BODY, retry_body(1), retry_body(2), retry_body(3)]
    assert [x["first_failed"] for x in body["attempts"]] == ["names", "parse", "duplicates", "parse"]
    assert [x["reward"] for x in body["attempts"]] == [pytest.approx(0.865), 0.0, pytest.approx(0.91), 0.0]


def test_all_fail_tie_keeps_the_earliest_greedy_included(start):
    client, _ = start(contents=(DUP, INVENTED, BROKEN, DUP))
    body = post(client).json()
    assert body["source"] == "model_greedy" and body["verified"] is False and body["model_calls"] == 4


def test_unparseable_best_answer_comes_back_raw(start):
    client, _ = start(contents=(BROKEN,), MAX_RETRIES="1")
    body = post(client).json()
    assert body["model_calls"] == 2 and body["verified"] is False
    assert body["workout"] is None and body["error"] is None and body["raw_output"] == BROKEN


@pytest.mark.parametrize("settings, calls, temperature", [({"MAX_RETRIES": "0"}, 1, None),
                                                           ({"MAX_RETRIES": "1", "RETRY_TEMPERATURE": "1.0"}, 2, 1.0)])
def test_retry_settings_from_env(start, settings, calls, temperature):
    client, server = start(contents=(DUP,), **settings)
    assert post(client).json()["model_calls"] == calls
    assert [r["temperature"] for r in server.requests[1:]] == ([temperature] if temperature else [])


@pytest.mark.parametrize("text, field", [
    ("medium workout with free weights", "missing"),
    ("chest and back workout, medium, free weights", "conflicts"),
    ("chest, medium, free weights, 7 exercises", "bad_count"),
    ("chest, medium, free weights, two exercises", "bad_count"),
])
def test_unreadable_request_is_422_with_the_allowed_words(start, text, field):
    client, server = start()
    r = post(client, text)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail[field] and "pecs" in detail["allowed"]["body_part"]["chest"]
    assert detail["example"] == EXAMPLE_REQUEST and EXAMPLE_REQUEST in detail["message"]
    assert server.requests == []


@pytest.mark.parametrize("text", ["", "x" * (api.MAX_REQUEST_CHARS + 1)])
def test_empty_or_too_long_request_is_422(start, text):
    client, server = start()
    assert post(client, text).status_code == 422 and server.requests == []


def test_forearms_is_not_arms(start):
    client, _ = start()
    body = post(client, "Forearms workout, hard, free weights, 3 exercises").json()
    assert body["request"]["body_part"] == "forearms"


def test_llama_server_down_is_503_but_the_guard_still_answers(start):
    client, server = start()
    server.stop()
    r = post(client)
    assert r.status_code == 503 and "llama-server request failed" in r.json()["detail"]
    assert post(client, "Forearms workout, easy difficulty, bands, 3 exercises").status_code == 200


def test_health(start):
    client, server = start()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_file": "model-Q8_0.gguf", "server_build": "b12360-fake"}
    server.stop()
    r = client.get("/health")
    assert r.status_code == 503 and r.json()["status"] == "unavailable"


def test_startup_fails_on_a_small_context(start):
    with pytest.raises(LlamaServerError, match="context per slot is 1024"):
        start(n_ctx=1024)


def test_startup_checks_the_expected_model_file(start):
    with pytest.raises(RuntimeError, match="has loaded 'model-Q8_0.gguf', not EXPECTED_MODEL_FILE 'other.gguf'"):
        start(EXPECTED_MODEL_FILE="other.gguf")


def test_startup_accepts_the_right_model_file(start):
    client, _ = start(EXPECTED_MODEL_FILE="model-Q8_0.gguf")
    assert client.get("/health").status_code == 200


def test_openapi_prefills_the_example_and_types_health(start):
    client, _ = start()
    spec = client.get("/openapi.json").json()
    examples = spec["paths"]["/workout"]["post"]["requestBody"]["content"]["application/json"]["examples"]
    assert examples["valid"]["value"] == {"request": EXAMPLE_REQUEST}
    health = spec["paths"]["/health"]["get"]["responses"]
    assert health["200"]["content"]["application/json"]["schema"] == {"$ref": "#/components/schemas/HealthResponse"}
    assert health["503"]["content"]["application/json"]["schema"] == {"$ref": "#/components/schemas/Unavailable"}
    assert {"interpreted_as", "model_input"} <= set(spec["components"]["schemas"]["WorkoutResponse"]["properties"])
