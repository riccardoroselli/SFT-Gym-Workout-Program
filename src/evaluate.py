"""Evaluate a model on the test set in two separate steps.

    python src/evaluate.py generate --run base_zeroshot              # model -> eval/outputs/base_zeroshot.jsonl
    python src/evaluate.py generate --run base_fewshot3 --few-shot 3
    python src/evaluate.py generate --run sft_grpo --model riccardoroselli/qwen3.5-2b-gym-grpo  # from the Hub (private: HF_TOKEN)
    python src/evaluate.py generate --run base_zeroshot --limit 5    # quick check on the first 5 examples
    python src/evaluate.py score --run base_zeroshot                 # -> eval/results/base_zeroshot.json + results.md

    # a GGUF file through a running llama-server (start it first: llama-server -m <file> -ngl all -c 4096 -np 1)
    python src/evaluate.py generate --run grpo_q8_0 --backend llamacpp --model models/gguf/qwen3.5-2b-gym-grpo-Q8_0.gguf

generate only saves raw answers; score only runs the verifier. Outputs made elsewhere
(e.g. Hugging Face on Colab) score the same way if they use the same jsonl format.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
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
DEFAULT_SERVER = "http://127.0.0.1:8080"
DEFAULT_TOKENIZER = "riccardoroselli/qwen3.5-2b-gym-grpo"  # build_prompt() for llamacpp needs the HF tokenizer
MIN_SERVER_CTX = 2048  # per slot: prompt (with few-shot turns) + MAX_NEW_TOKENS must fit
# Greedy, and every repetition penalty off whatever flags the server was started with, so the answers match mlx's.
GREEDY = {"temperature": 0, "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
          "dry_multiplier": 0.0}
STOP_TYPES = {"eos": False, "word": False, "limit": True}  # llama-server stop_type -> truncated
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


def mlx_backend(model_name, adapter_path):
    """-> (tokenizer, complete(prompt) -> (text, truncated, tokens, tps), extra row fields)."""
    from mlx_lm import load, stream_generate  # imported here so `score` works without mlx
    from mlx.utils import tree_flatten
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(model_name, adapter_path=adapter_path)
    dtypes = sorted({str(w.dtype) for _, w in tree_flatten(model.parameters())})
    print(f"loaded {model_name}, weight dtypes: {dtypes}")  # quantized weights would show as uint32
    greedy = make_sampler(temp=0.0)  # always the most likely token: same input, same output

    def complete(prompt):
        pieces = list(stream_generate(model, tokenizer, prompt, max_tokens=MAX_NEW_TOKENS, sampler=greedy))
        truncated = pieces[-1].finish_reason == "length"  # stopped by the token limit, not by the model
        # generation_tokens counts the <|im_end|> token too (the loop counts it, then stops)
        return "".join(r.text for r in pieces), truncated, pieces[-1].generation_tokens, round(pieces[-1].generation_tps, 1)
    return tokenizer, complete, {}


def http_json(url, body=None, timeout=600):
    """GET (body None) or POST a JSON body -> the parsed JSON answer."""
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def llamacpp_backend(model_path, adapter_path, server_url, tokenizer_name, tokenizer=None):
    """A running llama-server that must have loaded the .gguf file `model_path`. Same return as mlx_backend."""
    if adapter_path:
        sys.exit("--adapter-path works only with --backend mlx (merge the adapter before converting to GGUF)")
    url = server_url.rstrip("/")
    try:
        http_json(f"{url}/health", timeout=5)  # 503 while the model is still loading
    except (urllib.error.URLError, OSError) as e:
        sys.exit(f"llama-server not ready at {url}: {e}")
    props = http_json(f"{url}/props", timeout=5)

    # model_path is the server's -m argument: a relative one is relative to where the server was started
    loaded, wanted = Path(props.get("model_path", "")), Path(model_path)
    same = loaded.name == wanted.name and (not loaded.is_absolute() or (
        loaded.exists() and wanted.exists() and os.path.samefile(loaded, wanted)))
    if not same:
        sys.exit(f"llama-server at {url} has loaded {str(loaded)!r}, not --model {model_path!r}")
    n_ctx = props["default_generation_settings"]["n_ctx"]  # per slot: -np N splits -c by N
    if n_ctx < MIN_SERVER_CTX:
        sys.exit(f"llama-server context per slot is {n_ctx}, need at least {MIN_SERVER_CTX}: "
                 f"restart it with -c 4096 -np 1")
    build = props.get("build_info")
    print(f"llama-server {url}: {loaded}, build {build}, context per slot {n_ctx}")

    if tokenizer is None:
        from transformers import AutoTokenizer  # the prompt must be exactly build_prompt(), as with mlx
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)

    def complete(prompt):
        # raw prompt to /completion: no chat template on the server side
        out = http_json(f"{url}/completion", {"prompt": prompt, "n_predict": MAX_NEW_TOKENS, "stop": ["<|im_end|>"],
                                              "cache_prompt": False, **GREEDY})
        if out.get("stop_type") not in STOP_TYPES:
            sys.exit(f"llama-server returned stop_type {out.get('stop_type')!r}, expected one of {list(STOP_TYPES)}")
        # tokens_predicted counts the <|im_end|> token too, like mlx's generation_tokens
        return (out["content"], STOP_TYPES[out["stop_type"]], out["tokens_predicted"],
                round(out["timings"]["predicted_per_second"], 1))
    return tokenizer, complete, {"backend": "llamacpp", "model_file": wanted.name, "server_build": build}


def generate(run, model_name, adapter_path, k, limit=None, backend="mlx", server_url=DEFAULT_SERVER,
             tokenizer_name=DEFAULT_TOKENIZER, tokenizer=None):
    """tokenizer: only for llamacpp, to skip loading tokenizer_name (tests pass a fake one)."""
    if backend == "mlx":
        tokenizer, complete, extra = mlx_backend(model_name, adapter_path)
    else:
        tokenizer, complete, extra = llamacpp_backend(model_name, adapter_path, server_url, tokenizer_name, tokenizer)
    shots = few_shot_turns(k)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS / f"{run}.jsonl"
    test = read_jsonl(TEST)[:limit]
    start = time.time()
    with out_path.open("w") as f:
        for i, ex in enumerate(test, 1):
            system, user = ex["messages"][0], ex["messages"][1]  # drop the reference answer
            prompt = build_prompt(tokenizer, [system, *shots, user])
            text, truncated, tokens, tps = complete(prompt)
            row = {"meta": ex["meta"], "request": user["content"], "output": text, "truncated": truncated,
                   "tokens": tokens, "tps": tps, "model": model_name, "adapter": adapter_path, "few_shot": k, **extra}
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


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["generate", "score"])
    parser.add_argument("--run", required=True, help="name of the run, e.g. base or fewshot3")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="mlx: Hub repo or folder; llamacpp: the .gguf file")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--few-shot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="only the first N test examples")
    parser.add_argument("--backend", choices=["mlx", "llamacpp"], default="mlx")
    parser.add_argument("--server-url", default=DEFAULT_SERVER, help="llamacpp: the running llama-server")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="llamacpp: HF tokenizer for build_prompt()")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.step == "generate":
        generate(args.run, args.model, args.adapter_path, args.few_shot, args.limit,
                 args.backend, args.server_url, args.tokenizer)
    else:
        print(json.dumps(score(args.run), indent=1))
