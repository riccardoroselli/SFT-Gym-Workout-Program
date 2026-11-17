"""Tests for src/eval_service.py with a fake API: run (resume, meta check, non-200), derive (every K rule, ties), report."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import eval_service as es
from make_dataset import error_answer

NAMES = ["barbell bench press", "dumbbell bench press", "barbell incline bench press", "barbell decline bench press"]
GOOD = json.dumps({"workout": [{"name": n, "sets": 3, "reps": "8-12"} for n in NAMES]})          # reward 1.0
DUP = json.dumps({"workout": [{"name": n, "sets": 3, "reps": "8-12"} for n in NAMES[:3] + NAMES[:1]]})  # 0.91
CHEST = {"body_part": "chest", "equipment_group": "free weights", "difficulty": "medium", "n_exercises": 4, "feasible": True}
FOREARMS = {"body_part": "forearms", "equipment_group": "bands", "difficulty": "easy", "n_exercises": 3, "feasible": False}


def attempt(k, output, reward, ok, ms=1000, first_failed=None):
    return {"source": f"model_retry_{k}" if k else "model_greedy", "temperature": 0.7 if k else 0.0, "seed": k or None,
            "raw_output": output, "tokens": 30, "tps": 43.0, "truncated": False, "ms": ms, "reward": reward,
            "all_correct": ok, "first_failed": first_failed or ("passed" if ok else "duplicates")}


def response(meta, attempts, overhead=5):
    """What the API returns: the live answer follows the API's rule over all the attempts given."""
    if not attempts:
        source, output, model_ms = "guard", error_answer(meta), 0
    else:
        best = attempts[-1] if attempts[-1]["all_correct"] else max(attempts, key=lambda a: a["reward"])
        source, output, model_ms = best["source"], best["raw_output"], sum(a["ms"] for a in attempts)
    return {"request": dict(meta), "raw_output": output, "source": source, "model_input": "x",
            "verified": bool(not attempts or best["all_correct"]), "attempts": attempts, "model_calls": len(attempts),
            "timings_ms": {"model": model_ms, "total": model_ms + overhead},
            "model_file": "m-Q8_0.gguf", "server_build": "b-fake"}


def raw(meta, attempts, request="chest please"):
    return {"meta": meta, "request": request, "response": response(meta, attempts)}


# --- derive ---
def test_guard_row_is_the_error_answer_at_every_k():
    r = raw(FOREARMS, [])
    for k in range(4):
        row = es.system_row(r, k)
        assert (row["output"], row["source"], row["model_calls"], row["total_ms"]) == (error_answer(FOREARMS), "guard", 0, 5)
        assert row["truncated"] is False and row["meta"] == FOREARMS and row["max_retries"] == k


def test_pass_at_retry_2():
    r = raw(CHEST, [attempt(0, DUP, 0.91, False), attempt(1, "bad", 0.5, False), attempt(2, GOOD, 1.0, True)])
    got = [(row["source"], row["model_calls"], row["total_ms"], row["output"])
           for row in (es.system_row(r, k) for k in range(4))]
    assert got == [("model_greedy", 1, 1005, DUP),        # r0: greedy only
                   ("model_greedy", 2, 2005, DUP),        # r1: no pass, best reward of 2 calls is greedy
                   ("model_retry_2", 3, 3005, GOOD),      # r2: first pass
                   ("model_retry_2", 3, 3005, GOOD)]      # r3: stops at the pass, no 4th call


def test_all_fail_best_reward_and_earliest_on_ties():
    r = raw(CHEST, [attempt(0, "a", 0.8, False), attempt(1, "b", 0.9, False),
                    attempt(2, "c", 0.95, False), attempt(3, "d", 0.95, False)])
    assert [es.system_row(r, k)["output"] for k in range(4)] == ["a", "b", "c", "c"]  # c and d tie: c is earlier
    assert [es.system_row(r, k)["model_calls"] for k in range(4)] == [1, 2, 3, 4]


def test_greedy_pass_costs_one_call_at_every_k():
    r = raw(CHEST, [attempt(0, GOOD, 1.0, True, ms=1500)])
    assert {(es.system_row(r, k)["source"], es.system_row(r, k)["model_calls"], es.system_row(r, k)["total_ms"])
            for k in range(4)} == {("model_greedy", 1, 1505)}


def test_too_few_saved_attempts_is_refused():
    r = raw(CHEST, [attempt(0, DUP, 0.91, False), attempt(1, DUP, 0.91, False)])  # a run with MAX_RETRIES=1
    assert es.system_row(r, 1)["model_calls"] == 2
    with pytest.raises(AssertionError, match="r2 needs 3"):
        es.system_row(r, 2)


def test_derive_writes_four_files(tmp_path):
    raws = [raw(CHEST, [attempt(0, GOOD, 1.0, True)]), raw(FOREARMS, [])]
    (tmp_path / "s_raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raws))
    es.derive("s", tmp_path)
    for k in range(4):
        rows = [json.loads(line) for line in (tmp_path / f"s_r{k}.jsonl").read_text().splitlines()]
        assert [r["source"] for r in rows] == ["model_greedy", "guard"]


# --- run, against a fake API ---
class FakeAPI:
    """POST /workout: answers from `answers` (request text -> (status, body)); records the request texts."""

    def __init__(self, answers):
        self.answers, self.requests = answers, []
        api = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                text = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["request"]
                api.requests.append(text)
                status, body = api.answers[text]
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def api():
    servers = []
    yield lambda answers: servers.append(FakeAPI(answers)) or servers[-1]
    for s in servers:
        s.stop()


def write_test(path, rows):
    path.write_text("".join(json.dumps({"meta": m, "messages": [{"role": "system", "content": "s"},
                                                                 {"role": "user", "content": t}]}) + "\n"
                            for t, m in rows))


def test_run_saves_rows_and_resumes(tmp_path, api):
    test = tmp_path / "test.jsonl"
    write_test(test, [("chest one", CHEST), ("forearms", FOREARMS), ("chest two", CHEST)])
    fake = api({"chest one": (200, response(CHEST, [attempt(0, GOOD, 1.0, True)])),
                "forearms": (200, response(FOREARMS, [])),
                "chest two": (200, response(CHEST, [attempt(0, GOOD, 1.0, True)]))})
    es.run("s", limit=2, api_url=fake.url, outputs_dir=tmp_path, test_path=test)
    es.run("s", api_url=fake.url, outputs_dir=tmp_path, test_path=test)  # resume: only the third row is new
    assert fake.requests == ["chest one", "forearms", "chest two"]
    rows = [json.loads(line) for line in (tmp_path / "s_raw.jsonl").read_text().splitlines()]
    assert [(r["request"], r["meta"], r["response"]["source"]) for r in rows] == [
        ("chest one", CHEST, "model_greedy"), ("forearms", FOREARMS, "guard"), ("chest two", CHEST, "model_greedy")]


def test_run_refuses_a_saved_file_of_other_rows(tmp_path, api):
    test = tmp_path / "test.jsonl"
    write_test(test, [("chest one", CHEST)])
    (tmp_path / "s_raw.jsonl").write_text(json.dumps(raw(CHEST, [], request="something else")) + "\n")
    with pytest.raises(AssertionError, match="is not test row 0"):
        es.run("s", api_url=api({}).url, outputs_dir=tmp_path, test_path=test)


def test_run_stops_when_the_parsed_request_is_not_the_meta(tmp_path, api):
    test = tmp_path / "test.jsonl"
    write_test(test, [("chest one", CHEST)])
    fake = api({"chest one": (200, response({**CHEST, "n_exercises": 5}, [attempt(0, GOOD, 1.0, True)]))})
    with pytest.raises(AssertionError, match="parsed"):
        es.run("s", api_url=fake.url, outputs_dir=tmp_path, test_path=test)
    assert (tmp_path / "s_raw.jsonl").read_text() == ""  # nothing saved for that row


def test_run_stops_on_a_non_200(tmp_path, api):
    test = tmp_path / "test.jsonl"
    write_test(test, [("chest one", CHEST), ("chest two", CHEST)])
    fake = api({"chest one": (200, response(CHEST, [attempt(0, GOOD, 1.0, True)])),
                "chest two": (503, {"detail": "llama-server request failed"})})
    with pytest.raises(SystemExit, match="row 1: HTTP 503"):
        es.run("s", api_url=fake.url, outputs_dir=tmp_path, test_path=test)
    assert len((tmp_path / "s_raw.jsonl").read_text().splitlines()) == 1  # row 0 kept: a rerun resumes at row 1


# --- report ---
def test_report(tmp_path):
    raws = [raw(CHEST, [attempt(0, GOOD, 1.0, True, ms=1000)], "r0"),                                # greedy pass
            raw(CHEST, [attempt(0, DUP, 0.91, False, ms=1000), attempt(1, GOOD, 1.0, True, ms=2000)], "r1"),  # rescued
            raw(CHEST, [attempt(0, "x", 0.0, False, ms=500, first_failed="parse")] +
                [attempt(k, DUP, 0.91, False, ms=500) for k in (1, 2, 3)], "r2"),                     # still failing
            raw(FOREARMS, [], "r3")]                                                                  # guard
    (tmp_path / "s_raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raws))
    ref = [{"request": r["request"], "output": o} for r, o in zip(raws, [GOOD, "different", "x", "guard row"])]
    (tmp_path / "ref.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ref))
    out = es.report("s", tmp_path, reference="ref")
    assert out["greedy"] == {"rows": 3, "identical": 2, "differ": [1]}  # the guard row is not compared
    k = out["per_k"]
    assert [k[i]["all_correct"] for i in range(4)] == [0.5, 0.75, 0.75, 0.75]
    assert [k[i]["rescued"] for i in range(4)] == [0, 1, 1, 1]
    assert [k[i]["calls_per_row"] for i in range(4)] == [0.75, 1.25, 1.5, 1.75]
    assert k[3]["calls_per_model_row"] == round(7 / 3, 3)
    assert k[0]["reward"] == round((1.0 + 0.91 + 0.0 + 1.0) / 4, 4)
    assert out["latency"]["guard"] == {"rows": 1, "mean": 5, "p50": 5, "p95": 5}
    # the rescued row (3005 ms) and the still-failing one, whose tied best reward is retry 1 too (2005 ms)
    assert out["latency"]["model_retry_1"] == {"rows": 2, "mean": 2505, "p50": 2005, "p95": 3005}
    assert out["greedy_failures"] == {"rescued": {"duplicates": 1}, "still failing": {"parse": 1}}


def test_percentile_is_nearest_rank():
    assert es.percentile(list(range(1, 101)), 95) == 95 and es.percentile([7], 50) == 7
    assert es.percentile([1, 2, 3, 4], 50) == 2
