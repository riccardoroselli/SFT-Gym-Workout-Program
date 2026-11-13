"""Paired comparison of two evaluated runs on the same test requests. Reads eval/outputs/, prints, writes nothing.

    python src/compare_runs.py sft_grpo sft_grpo_gguf_bf16          # A -> B: is B better or worse than A?

Both runs answered the same requests in the same order, so each request is compared with itself:
- answers with identical text;
- among the differing ones: B's reward higher / lower / equal, all_correct gained / lost;
- mean difference B - A of reward and all_correct, with a 95% paired bootstrap interval
  (resample the requests 10,000 times, seed 42; the same resamples for both). If it contains 0, it's noise;
- false refusals: {"error": ...} answers to feasible requests.
"""
import argparse
import json

import numpy as np

from evaluate import OUTPUTS, read_jsonl
from verifier import verify

RESAMPLES = 10_000
SEED = 42


def is_refusal(output):
    try:
        obj = json.loads(output)
    except json.JSONDecodeError:
        return False
    return isinstance(obj, dict) and "error" in obj


def compare(rows_a, rows_b, resamples=RESAMPLES, seed=SEED):
    """Two lists of evaluate.py output rows -> dict of paired statistics (B relative to A)."""
    if [r["request"] for r in rows_a] != [r["request"] for r in rows_b]:
        raise ValueError("the two runs did not answer the same requests in the same order")
    res_a = [verify(r["meta"], r["output"]) for r in rows_a]
    res_b = [verify(r["meta"], r["output"]) for r in rows_b]
    reward_a, reward_b = (np.array([x["reward"] for x in res]) for res in (res_a, res_b))
    correct_a, correct_b = (np.array([float(x["all_correct"]) for x in res]) for res in (res_a, res_b))
    differ = np.array([a["output"] != b["output"] for a, b in zip(rows_a, rows_b)])
    d_reward, d_correct = reward_b - reward_a, correct_b - correct_a
    eps = 1e-9  # rewards are sums of floats: equal within rounding counts as equal
    higher, lower = int((d_reward > eps)[differ].sum()), int((d_reward < -eps)[differ].sum())

    idx = np.random.default_rng(seed).integers(0, len(rows_a), size=(resamples, len(rows_a)))

    def interval(d):
        means = d[idx].mean(axis=1)
        return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    def false_refusals(rows):
        return sum(is_refusal(r["output"]) for r in rows if r["meta"]["feasible"])

    return {"n": len(rows_a), "identical": int((~differ).sum()), "higher": higher, "lower": lower,
            "equal": int(differ.sum()) - higher - lower,
            "gained": int(((correct_b == 1) & (correct_a == 0)).sum()),
            "lost": int(((correct_a == 1) & (correct_b == 0)).sum()),
            "reward": interval(d_reward), "all_correct": interval(d_correct),
            "false_refusals": (false_refusals(rows_a), false_refusals(rows_b)),
            "feasible": sum(r["meta"]["feasible"] for r in rows_a)}


def report(run_a, run_b):
    s = compare(read_jsonl(OUTPUTS / f"{run_a}.jsonl"), read_jsonl(OUTPUTS / f"{run_b}.jsonl"))
    fmt = lambda t, digits: f"{t[0]:+.{digits}f} [{t[1]:+.{digits}f}, {t[2]:+.{digits}f}]"
    print(f"== {run_a} -> {run_b}: {s['n']} requests")
    print(f"   identical answers:         {s['identical']}/{s['n']}")
    print(f"   differing, B reward:       higher {s['higher']} / lower {s['lower']} / equal {s['equal']}")
    print(f"   all_correct:               gained {s['gained']} / lost {s['lost']}")
    print(f"   delta reward [95% CI]:      {fmt(s['reward'], 4)}")
    print(f"   delta all_correct [95% CI]: {fmt(s['all_correct'], 3)}")
    print(f"   false refusals (of {s['feasible']} feasible): A {s['false_refusals'][0]}, B {s['false_refusals'][1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a", help="reference run: a file in eval/outputs/, without .jsonl")
    parser.add_argument("run_b", help="run compared with it")
    args = parser.parse_args()
    report(args.run_a, args.run_b)
