"""Before GRPO: does sampling the SFT model give a learning signal?

GRPO samples G answers per request and learns only from groups whose rewards differ: if all G answers
get the same reward (zero std), the group teaches nothing. This samples G answers for 40 fixed train
requests, scores each with verify(), and prints per category how much signal there is.

    python src/probe_groups.py --run sft_t1                        # 40 requests x 8 answers, temperature 1.0
    python src/probe_groups.py --run smoke --limit 2 --group-size 2

Reads train.jsonl, writes only eval/outputs/probe_<run>.jsonl (one line per answer, flushed after each group).
"""
import argparse
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict

from evaluate import OUTPUTS, TRAIN, build_prompt, read_jsonl
from verifier import verify

SEED = 42
DEFAULT_MODEL = "riccardoroselli/qwen3.5-2b-gym-sft"
MAX_NEW_TOKENS = 256
N_IMPOSSIBLE, N_PER_COUNT = 8, 8  # 8 impossible + 8 feasible for each count 3/4/5/6 = 40 requests


def combo(meta):
    return meta["body_part"], meta["equipment_group"], meta["difficulty"], meta["n_exercises"]


def pick_requests(rows, seed=SEED):
    """40 train rows with distinct combos: 8 impossible covering every impossible (body part, equipment) pair,
    then 8 feasible per count, spread over equipment groups and difficulties. Fixed seed, shuffled order."""
    rng = random.Random(seed)
    unique = list({combo(r["meta"]): r for r in rows}.values())  # one row per combo
    rng.shuffle(unique)

    impossible = [r for r in unique if not r["meta"]["feasible"]]
    by_pair = {}
    for r in impossible:
        by_pair.setdefault(combo(r["meta"])[:2], r)
    chosen = list(by_pair.values())  # one per pair first
    chosen += [r for r in impossible if r not in chosen][:N_IMPOSSIBLE - len(chosen)]

    for n in (3, 4, 5, 6):
        pool = [r for r in unique if r["meta"]["feasible"] and r["meta"]["n_exercises"] == n]
        used = Counter()
        for _ in range(N_PER_COUNT):  # greedy: the least-used equipment group + difficulty next
            best = min(pool, key=lambda r: used[r["meta"]["equipment_group"]] + used[r["meta"]["difficulty"]])
            pool.remove(best)
            chosen.append(best)
            used.update([best["meta"]["equipment_group"], best["meta"]["difficulty"]])
    rng.shuffle(chosen)
    return chosen


def category(meta):
    return "impossible" if not meta["feasible"] else f"feasible n={meta['n_exercises']}"


def summarize(rows):
    """rows: scored answers ({"idx", "meta", "tokens", "truncated", "result"}) -> {category: stats}."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["idx"]].append(r)
    cats = defaultdict(list)
    for g in groups.values():
        cats[category(g[0]["meta"])].append(g)
        cats["all"].append(g)

    out = {}
    for cat, gs in cats.items():
        answers = [a for g in gs for a in g]
        res = [a["result"] for a in answers]
        stds = [statistics.pstdev(a["result"]["reward"] for a in g) for g in gs]
        refused = [r["parse_ok"] and r["names_exist"] is None for r in res]  # a parsed answer that isn't a workout
        workouts = [r for r in res if r["no_duplicates"] is not None]
        lengths = sorted(a["tokens"] for a in answers)
        feasible = [a["meta"]["feasible"] for a in answers]
        rate = lambda flags: round(sum(flags) / len(flags), 4) if flags else None
        out[cat] = {
            "groups": len(gs), "answers": len(answers),
            "reward": round(statistics.mean(r["reward"] for r in res), 4),
            "reward_std": round(statistics.mean(stds), 4),
            "zero_std": rate([s < 1e-9 for s in stds]),
            "all_correct": rate([bool(r["all_correct"]) for r in res]),
            "pass_at_g": rate([any(a["result"]["all_correct"] for a in g) for g in gs]),
            "refusal": rate([x for x, f in zip(refused, feasible) if not f]),        # impossible: wanted
            "false_refusal": rate([x for x, f in zip(refused, feasible) if f]),     # feasible: wrong
            "duplicates": rate([not r["no_duplicates"] for r in workouts]),
            "tokens_max": lengths[-1],
            "tokens_p95": lengths[math.ceil(0.95 * len(lengths)) - 1],  # nearest-rank percentile
            "hit_limit": sum(bool(a["truncated"]) for a in answers),
        }
    return out


def print_summary(stats, g):
    order = ["impossible", *(f"feasible n={n}" for n in (3, 4, 5, 6)), "all"]
    cols = ["groups", "answers", "reward", "reward_std", "zero_std", "all_correct", "pass_at_g",
            "refusal", "false_refusal", "duplicates", "tokens_max", "tokens_p95", "hit_limit"]
    print(f"\n| category | {' | '.join(c.replace('pass_at_g', f'pass@{g}') for c in cols)} |")
    print("|" + "---|" * (len(cols) + 1))
    for cat in [c for c in order if c in stats]:
        print(f"| {cat} | " + " | ".join("-" if stats[cat][c] is None else str(stats[cat][c]) for c in cols) + " |")


def encode(tokenizer, prompt):
    """Same token ids as stream_generate() gets from evaluate.py's text prompt."""
    bos = tokenizer.bos_token
    return tokenizer.encode(prompt, add_special_tokens=bos is None or not prompt.startswith(bos))


def sample_group(model, tokenizer, prompt, g, sampler):
    """-> [(text, n_tokens, hit_limit)] for g sampled answers to one prompt."""
    import mlx_lm
    ids = encode(tokenizer, prompt)
    if hasattr(mlx_lm, "batch_generate"):  # all g answers in one batch
        resp = mlx_lm.batch_generate(model, tokenizer, [ids] * g, max_tokens=MAX_NEW_TOKENS,
                                     sampler=sampler, return_token_ids=True)
        # the stop token is not kept, so only an answer cut by the limit has MAX_NEW_TOKENS tokens
        return [(t, len(tok), len(tok) >= MAX_NEW_TOKENS) for t, tok in zip(resp.texts, resp.token_ids)]
    out = []  # older mlx-lm: one answer at a time
    for _ in range(g):
        pieces = list(mlx_lm.stream_generate(model, tokenizer, ids, max_tokens=MAX_NEW_TOKENS, sampler=sampler))
        out.append(("".join(p.text for p in pieces), pieces[-1].generation_tokens, pieces[-1].finish_reason == "length"))
    return out


def generate(run, model_name, g, temperature, limit):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(model_name)
    sampler = make_sampler(temp=temperature)
    mx.random.seed(SEED)
    requests = pick_requests(read_jsonl(TRAIN))[:limit]
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS / f"probe_{run}.jsonl"
    start = time.time()
    with path.open("w") as f:
        for i, ex in enumerate(requests):
            system, user = ex["messages"][0], ex["messages"][1]  # drop the reference answer
            t0 = time.time()
            answers = sample_group(model, tokenizer, build_prompt(tokenizer, [system, user]), g, sampler)
            for j, (text, n_tokens, hit) in enumerate(answers):
                f.write(json.dumps({"idx": i, "sample": j, "meta": ex["meta"], "request": user["content"],
                                    "output": text, "tokens": n_tokens, "truncated": hit, "model": model_name,
                                    "temperature": temperature}, ensure_ascii=False) + "\n")
            f.flush()
            dt = time.time() - t0
            print(f"[{i + 1}/{len(requests)}] {time.time() - start:.0f}s  group {dt:.1f}s, "
                  f"{sum(a[1] for a in answers) / dt:.1f} tok/s  {category(ex['meta'])}", flush=True)
    print(f"saved {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="output name: eval/outputs/probe_<run>.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--group-size", type=int, default=8, help="answers per request (G)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None, help="only the first N of the 40 requests")
    args = parser.parse_args()
    path = generate(args.run, args.model, args.group_size, args.temperature, args.limit)
    rows = [{**r, "result": verify(r["meta"], r["output"])} for r in read_jsonl(path)]
    print_summary(summarize(rows), args.group_size)
