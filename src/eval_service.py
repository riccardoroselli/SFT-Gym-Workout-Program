"""Evaluate the whole service (parser + guard + model + verifier retries) on the test set, through its HTTP API.

    python src/eval_service.py run --run system_q8 [--limit N]   # POST every test request -> eval/outputs/system_q8_raw.jsonl
    python src/eval_service.py derive --run system_q8            # -> system_q8_r0..r3.jsonl, evaluate.py's row format
    python src/eval_service.py report --run system_q8            # greedy vs the Q8_0 run, retries, latency, failed checks
    python src/evaluate.py score --run system_q8_r3              # score a derived run like any other

run needs the API (uvicorn, MAX_RETRIES=3) and its llama-server running. It saves each full response as soon as it
arrives and skips rows already saved, so it can be stopped and started again. Any non-200 answer stops it.
derive works offline: rK is the answer the service would give with MAX_RETRIES=K, rebuilt from the saved attempts
with the API's own rule (the first all-correct attempt, else the best reward, earliest on a tie).
Standard library only, like llama_client.py.
"""
import argparse
import collections
import json
import math
import sys
import urllib.error
from pathlib import Path

from evaluate import OUTPUTS, TEST, read_jsonl
from llama_client import http_json
from verifier import verify

DEFAULT_API = "http://127.0.0.1:8000"
MAX_K = 3  # r0..r3: the API's default MAX_RETRIES
REFERENCE = "sft_grpo_gguf_q8"  # evaluate.py's greedy run of the same GGUF through the same llama-server
KEYS = ["body_part", "equipment_group", "difficulty", "n_exercises"]


def run(name, limit=None, api_url=DEFAULT_API, outputs_dir=OUTPUTS, test_path=TEST):
    test = read_jsonl(test_path)[:limit]
    path = Path(outputs_dir) / f"{name}_raw.jsonl"
    done = read_jsonl(path) if path.exists() else []
    for i, row in enumerate(done):  # resuming: the saved rows must be the same test rows, in order
        assert row["request"] == test[i]["messages"][1]["content"], f"{path} row {i} is not test row {i}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for i in range(len(done), len(test)):
            meta, text = test[i]["meta"], test[i]["messages"][1]["content"]
            try:
                response = http_json(f"{api_url.rstrip('/')}/workout", {"request": text})
            except urllib.error.HTTPError as e:
                sys.exit(f"row {i}: HTTP {e.code} for {text!r}: {e.read().decode()[:500]}")
            parsed = {k: response["request"][k] for k in KEYS}
            assert parsed == {k: meta[k] for k in KEYS}, f"row {i}: parsed {parsed} != meta {meta} for {text!r}"
            f.write(json.dumps({"meta": meta, "request": text, "response": response}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(test)}] {response['source']:<14} {response['timings_ms']['total']:>6} ms  "
                  f"verified={response['verified']}", flush=True)
    print(f"saved {path}")


def at_k(response, k):
    """-> (chosen attempt or None for the guard, model calls, total ms) the service gives with MAX_RETRIES=k."""
    attempts = response["attempts"]
    if not attempts:
        return None, 0, response["timings_ms"]["total"]
    first_pass = next((i for i, a in enumerate(attempts[:k + 1]) if a["all_correct"]), None)
    # without a pass in the first k+1 attempts, the saved run must have made all k+1 of them
    assert first_pass is not None or len(attempts) >= k + 1, f"only {len(attempts)} attempts saved, r{k} needs {k + 1}"
    calls = first_pass + 1 if first_pass is not None else k + 1
    used = attempts[:calls]
    best = used[-1] if first_pass is not None else max(used, key=lambda a: a["reward"])  # max(): earliest on a tie
    overhead = response["timings_ms"]["total"] - response["timings_ms"]["model"]  # parsing, verifying, HTTP
    return best, calls, overhead + sum(a["ms"] for a in used)


def system_row(raw, k):
    """evaluate.py's row format (meta, request, output, truncated, tokens, tps, ...) plus source, model_calls, total_ms."""
    response = raw["response"]
    best, calls, total_ms = at_k(response, k)
    if best is not None and len(response["attempts"]) <= k + 1:  # self-check: with every attempt in reach, rk is the live answer
        assert best["source"] == response["source"], (raw["request"], k)
    chosen = best or {"source": "guard", "raw_output": response["raw_output"], "truncated": False, "tokens": 0, "tps": None}
    return {"meta": raw["meta"], "request": raw["request"], "output": chosen["raw_output"],
            "truncated": chosen["truncated"], "tokens": chosen["tokens"], "tps": chosen["tps"],
            "model_input": response["model_input"], "model_file": response["model_file"],
            "server_build": response["server_build"], "max_retries": k,
            "source": chosen["source"], "model_calls": calls, "total_ms": total_ms}


def derive(name, outputs_dir=OUTPUTS):
    raws = read_jsonl(Path(outputs_dir) / f"{name}_raw.jsonl")
    for k in range(MAX_K + 1):
        path = Path(outputs_dir) / f"{name}_r{k}.jsonl"
        path.write_text("".join(json.dumps(system_row(r, k), ensure_ascii=False) + "\n" for r in raws))
        print(f"saved {path} ({len(raws)} rows)")


def percentile(values, p):
    """Nearest rank: the smallest value with at least p% of the values at or below it."""
    values = sorted(values)
    return values[max(math.ceil(p / 100 * len(values)) - 1, 0)]


def report(name, outputs_dir=OUTPUTS, reference=REFERENCE):
    """Prints four tables and returns the same numbers as a dict (for the tests)."""
    raws = read_jsonl(Path(outputs_dir) / f"{name}_raw.jsonl")
    ref = read_jsonl(Path(outputs_dir) / f"{reference}.jsonl")
    out = {}

    # (a) the service's greedy call must be evaluate.py's llamacpp call: same prompt, same settings, same text
    feasible = [(i, r) for i, r in enumerate(raws) if r["response"]["attempts"]]
    differ = []
    for i, r in feasible:
        assert ref[i]["request"] == r["request"], f"row {i}: {reference} has another request"
        if r["response"]["attempts"][0]["raw_output"] != ref[i]["output"]:
            differ.append(i)
    out["greedy"] = {"rows": len(feasible), "identical": len(feasible) - len(differ), "differ": differ}
    print(f"(a) greedy text vs {reference}: {len(feasible) - len(differ)} / {len(feasible)} model rows identical")
    for i in differ:
        print(f"    row {i}: {raws[i]['request']!r}\n      service: {raws[i]['response']['attempts'][0]['raw_output'][:150]!r}"
              f"\n      {reference}: {ref[i]['output'][:150]!r}")

    # (b) what each retry budget buys; all_correct and reward recomputed exactly as evaluate.py score does
    greedy_ok = [verify(r["meta"], system_row(r, 0)["output"])["all_correct"] for r in raws]
    print(f"\n(b) {'K':>2} {'all_correct':>12} {'reward':>8} {'rescued':>8} {'calls/row':>10} {'calls/model row':>16}")
    out["per_k"] = {}
    for k in range(MAX_K + 1):
        rows = [system_row(r, k) for r in raws]
        results = [verify(r["meta"], r["output"]) for r in rows]
        ok = [bool(x["all_correct"]) for x in results]
        calls = [r["model_calls"] for r in rows]
        stats = {"all_correct": round(sum(ok) / len(ok), 4),
                 "reward": round(sum(x["reward"] for x in results) / len(results), 4),
                 "rescued": sum(o and not g for o, g in zip(ok, greedy_ok)),
                 "calls_per_row": round(sum(calls) / len(calls), 3),
                 "calls_per_model_row": round(sum(calls) / max(len(feasible), 1), 3)}
        out["per_k"][k] = stats
        print(f"    {k:>2} {stats['all_correct']:>12} {stats['reward']:>8} {stats['rescued']:>8} "
              f"{stats['calls_per_row']:>10} {stats['calls_per_model_row']:>16}")

    # (c) latency of the live answers (the API ran with MAX_RETRIES=3), by the source that answered and whether it
    # was verified: an unverified "model_greedy" is a row where all 4 calls failed and greedy had the best reward
    by_source = collections.defaultdict(list)
    for r in raws:
        by_source[r["response"]["source"], r["response"]["verified"]].append(r["response"]["timings_ms"]["total"])
    print(f"\n(c) {'source':<14} {'verified':<9} {'rows':>5} {'mean ms':>8} {'p50':>7} {'p95':>7}")
    out["latency"] = {}
    for source, verified in sorted(by_source, key=lambda sv: (sv[0] != "guard", sv[0], not sv[1])):
        ms = by_source[source, verified]
        s = out["latency"][source, verified] = {"rows": len(ms), "mean": round(sum(ms) / len(ms)),
                                                "p50": percentile(ms, 50), "p95": percentile(ms, 95)}
        print(f"    {source:<14} {str(verified).lower():<9} {s['rows']:>5} {s['mean']:>8} {s['p50']:>7} {s['p95']:>7}")

    # (d) the greedy failure behind each row that retries rescued, or didn't
    failed = [r for r in raws if r["response"]["attempts"] and not r["response"]["attempts"][0]["all_correct"]]
    out["greedy_failures"] = {
        outcome: dict(collections.Counter(r["response"]["attempts"][0]["first_failed"] for r in failed
                                          if r["response"]["verified"] == (outcome == "rescued")).most_common())
        for outcome in ("rescued", "still failing")}
    print(f"\n(d) first failed check on greedy, rows the retries (up to {MAX_K}) rescued vs still failing")
    for outcome, counts in out["greedy_failures"].items():
        print(f"    {outcome:<14} {sum(counts.values()):>4}: {counts}")
    return out


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["run", "derive", "report"])
    parser.add_argument("--run", required=True, help="name of the run, e.g. system_q8")
    parser.add_argument("--limit", type=int, default=None, help="run: only the first N test examples")
    parser.add_argument("--api-url", default=DEFAULT_API)
    parser.add_argument("--reference", default=REFERENCE, help="report: evaluate.py run to compare greedy text with")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.step == "run":
        run(args.run, args.limit, args.api_url)
    elif args.step == "derive":
        derive(args.run)
    else:
        report(args.run, reference=args.reference)
