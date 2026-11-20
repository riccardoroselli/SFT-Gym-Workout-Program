"""Tests for the web page at GET /: served files, no outside requests, chips that write the canonical request,
and the degree-sign fix being display-only (the API answer is unchanged)."""
import itertools
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api
from fakes import FakeServer, FakeTokenizer
from make_dataset import BODY_WORDS, EQUIP_WORDS, LEVEL_WORDS
from request_parser import canonical_text, parse_request
from verifier import is_feasible

WEB = Path(api.__file__).resolve().parent / "web"
HTML = (WEB / "index.html").read_text()
ALLOWED_LINKS = ("https://github.com/riccardoroselli/", "https://huggingface.co/riccardoroselli/")
client = TestClient(api.app)  # not entered as a context manager: no startup checks, no llama-server needed


def test_page_and_static_files_are_served():
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html") and "<h1>GymLeader</h1>" in r.text
    assert client.get("/static/app.js").headers["content-type"].startswith("text/javascript")
    assert client.get("/static/style.css").headers["content-type"].startswith("text/css")
    assert client.get("/static/nothing.js").status_code == 404


def test_api_contract_is_unchanged():
    assert set(client.get("/openapi.json").json()["paths"]) == {"/workout", "/health"}


def test_the_page_makes_no_outside_requests():
    assert "default-src 'self'" in HTML  # the browser enforces it (Content-Security-Policy)
    for path in WEB.iterdir():
        text = path.read_text()
        urls = re.findall(r"https?://[^\s\"'<>)]+", text)
        if path.name != "index.html":
            assert urls == [], f"{path.name} has outside URLs: {urls}"
            continue
        for url in urls:  # only plain links a person clicks, never something the page loads
            assert url.startswith(ALLOWED_LINKS) and f'<a href="{url}"' in text, url
    assert not re.search(r"""(?:src|href)=["']//""", HTML), "protocol-relative URL"
    assert not re.search(r"@import|url\(", (WEB / "style.css").read_text())


def chips():
    """-> {radio group name: [values]} from index.html."""
    groups = {}
    for name, value in re.findall(r'<input type="radio" name="(\w+)" value="([^"]+)"', HTML):
        groups.setdefault(name, []).append(value)
    return groups


def test_chips_write_the_canonical_sentence():
    groups = chips()
    assert groups == {"body": [friendly[0] for friendly, _ in BODY_WORDS.values()],
                      "equip": [words[0] for words in EQUIP_WORDS.values()],
                      "level": [words[0] for words in LEVEL_WORDS.values()],
                      "n": ["3", "4", "5", "6"]}
    assert re.search(r'name="n" value="4" checked', HTML)  # default count, as in the spec
    template = re.search(r'data-template="([^"]+)"', HTML).group(1)
    names = dict(zip(groups["body"], BODY_WORDS)), dict(zip(groups["equip"], EQUIP_WORDS))
    for body, equip, level, n in itertools.product(*groups.values()):  # what app.js does, for all 480 combinations
        text = template.replace("{body}", body).replace("{level}", level).replace("{equip}", equip).replace("{n}", n)
        text = text[0].upper() + text[1:]
        request = {"body_part": names[0][body], "equipment_group": names[1][equip], "difficulty": level, "n_exercises": int(n)}
        assert text == canonical_text(request)
        assert parse_request(text) == (request, [])


def test_examples_parse_and_one_is_impossible():
    examples = re.findall(r'<button type="button" class="example">([^<]+)</button>', HTML)
    assert len(examples) == 4
    feasible = [is_feasible(parse_request(e)[0]) for e in examples]
    assert feasible == [True, True, True, False]
    assert parse_request(examples[1])[1], "one example should show the interpreted_as notes"


def test_degree_fix_is_display_only(monkeypatch, tmp_path):
    name = "sled 45в° leg press"  # mis-encoded upstream; data-v1 and the verifier use it as it is
    assert name in (Path(api.__file__).resolve().parent.parent / "data/processed/catalog.json").read_text()
    output = json.dumps({"workout": [{"name": name, "sets": 4, "reps": "5-8"}]}, ensure_ascii=False)  # as the model writes it
    for env in ["EXPECTED_MODEL_FILE", "TOKENIZER", "RETRY_TEMPERATURE"]:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(api, "load_tokenizer", lambda name: FakeTokenizer())
    server = FakeServer(tmp_path / "model-Q8_0.gguf", contents=(output,))
    monkeypatch.setenv("LLAMA_SERVER_URL", server.url)
    monkeypatch.setenv("MAX_RETRIES", "0")
    try:
        with TestClient(api.app) as c:
            body = c.post("/workout", json={"request": "Legs workout, hard difficulty, machines, 3 exercises"}).json()
    finally:
        server.stop()
    assert body["workout"][0]["name"] == name and name in body["raw_output"]  # the API answer is unchanged
    assert 'replaceAll("в°", "°")' in (WEB / "app.js").read_text()  # the page fixes it for display only


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_source_badge_matches_the_model_calls():
    """app.js sourceText(), run in node: an unverified answer after several calls is 'best of N', even if greedy."""
    js = re.search(r"function sourceText\(r\) \{.*?\n\}", (WEB / "app.js").read_text(), re.DOTALL).group()
    cases = [({"source": "guard", "verified": True, "model_calls": 0}, "answered from the catalog"),
             ({"source": "model_greedy", "verified": True, "model_calls": 1}, "model, first answer"),
             ({"source": "model_greedy", "verified": False, "model_calls": 1}, "model, first answer"),  # MAX_RETRIES=0
             ({"source": "model_greedy", "verified": False, "model_calls": 4}, "model, best of 4 answers"),
             ({"source": "model_retry_2", "verified": False, "model_calls": 4}, "model, best of 4 answers"),
             ({"source": "model_retry_1", "verified": True, "model_calls": 2}, "model, rescued at retry 1")]
    script = js + f"\nconsole.log(JSON.stringify({json.dumps([c for c, _ in cases])}.map(sourceText)));"
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == [want for _, want in cases]
