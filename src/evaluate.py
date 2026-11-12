"""Evaluate a model on the test set in two separate steps.

    python src/evaluate.py generate --run base_zeroshot              # model -> eval/outputs/base_zeroshot.jsonl
    python src/evaluate.py generate --run base_fewshot3 --few-shot 3
    python src/evaluate.py generate --run sft_grpo --model riccardoroselli/qwen3.5-2b-gym-grpo  # from the Hub (private: HF_TOKEN)
    python src/evaluate.py generate --run base_zeroshot --limit 5    # quick check on the first 5 examples
    python src/evaluate.py score --run base_zeroshot                 # -> eval/results/base_zeroshot.json + results.md

generate only saves raw answers; score only runs the verifier. Outputs made elsewhere
(e.g. Hugging Face on Colab) score the same way if they use the same jsonl format.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from verifier import verify

ROOT = Path(__file__).resolve().parent.parent
TEST = ROOT / "data/processed/test.jsonl"
TRAIN = ROOT / "data/processed/train.jsonl"
OUTPUTS = ROOT / "eval/outputs"
RESULTS = ROOT / "eval/results"
DEFAULT_MODEL = "Qwen/Qwen3.5-2B"
MAX_NEW_TOKENS = 512
# Fixed train.jsonl rows for few-shot: medium, hard, then an impossible easy request (error answer).
FEW_SHOT_IDS = [2, 3, 15, 13, 7]
FIELDS = ["parse_ok", "strict_format", "names_exist", "body_part_ok", "equipment_ok", "count_ok",
          "no_duplicates", "sets_reps_ok", "advanced_ok", "error_correct", "all_correct", "reward", "has_think", "truncated"]


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def few_shot_turns(k):
    if k > len(FEW_SHOT_IDS):
        sys.exit(f"--few-shot supports at most {len(FEW_SHOT_IDS)} examples")
    train = read_jsonl(TRAIN)
    # user + assistant turns only; the system message is given once, by the test example
    return [m for i in FEW_SHOT_IDS[:k] for m in train[i]["messages"] if m["role"] != "system"]


def build_prompt(tokenizer, messages):
    """Chat messages -> prompt text, thinking off. The SFT notebook imports this so training matches eval."""
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)


def generate(run, model_name, adapter_path, k, limit=None):
    from mlx_lm import load, stream_generate  # imported here so `score` works without mlx
    from mlx.utils import tree_flatten
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(model_name, adapter_path=adapter_path)
    dtypes = sorted({str(w.dtype) for _, w in tree_flatten(model.parameters())})
    print(f"loaded {model_name}, weight dtypes: {dtypes}")  # quantized weights would show as uint32
    greedy = make_sampler(temp=0.0)  # always the most likely token: same input, same output
    shots = few_shot_turns(k)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS / f"{run}.jsonl"
    test = read_jsonl(TEST)[:limit]
    start = time.time()
    with out_path.open("w") as f:
        for i, ex in enumerate(test, 1):
            system, user = ex["messages"][0], ex["messages"][1]  # drop the reference answer
            prompt = build_prompt(tokenizer, [system, *shots, user])
            pieces = list(stream_generate(model, tokenizer, prompt, max_tokens=MAX_NEW_TOKENS, sampler=greedy))
            text = "".join(r.text for r in pieces)
            truncated = pieces[-1].finish_reason == "length"  # stopped by the token limit, not by the model
            row = {"meta": ex["meta"], "request": user["content"], "output": text, "truncated": truncated,
                   "tokens": pieces[-1].generation_tokens, "tps": round(pieces[-1].generation_tps, 1),
                   "model": model_name, "adapter": adapter_path, "few_shot": k}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(test)}] {time.time() - start:.0f}s  {text[:80]!r}", flush=True)
    print(f"saved {out_path}")


def mean(values):
    values = [float(v) for v in values if v is not None]  # None = check doesn't apply to this example
    return round(sum(values) / len(values), 4) if values else None


def score(run, outputs_dir=OUTPUTS, results_dir=RESULTS):
    rows = read_jsonl(Path(outputs_dir) / f"{run}.jsonl")
    per_example = []
    for row in rows:
        result = verify(row["meta"], row["output"])
        result["has_think"] = "<think>" in row["output"]
        result["truncated"] = row.get("truncated")  # None if the outputs file doesn't record it
        per_example.append({"meta": row["meta"], "output": row["output"], **result})
    summary = {"run": run, "n": len(rows), **{k: mean(r[k] for r in per_example) for k in FIELDS}}

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{run}.json").write_text(json.dumps({"summary": summary, "examples": per_example}, indent=1))
    update_table(results_dir / "results.md", summary)
    return summary


def update_table(path, summary):
    """One row per run: re-scoring a run replaces its row. Text lines under the table stay under it."""
    cols = ["run", "n", *FIELDS]
    header = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines = path.read_text().splitlines()[2:] if path.exists() else []
    rows = [line for line in lines if line.startswith("|") and not line.startswith(f"| {summary['run']} |")]
    notes = [line for line in lines if not line.startswith("|")]
    cells = ["-" if summary[c] is None else str(summary[c]) for c in cols]
    rows.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(header + rows + notes) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["generate", "score"])
    parser.add_argument("--run", required=True, help="name of the run, e.g. base or fewshot3")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--few-shot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="only the first N test examples")
    args = parser.parse_args()
    if args.step == "generate":
        generate(args.run, args.model, args.adapter_path, args.few_shot, args.limit)
    else:
        print(json.dumps(score(args.run), indent=1))
