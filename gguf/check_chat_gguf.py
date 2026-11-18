"""Checks for the chat-ready GGUF (gguf/make_chat_gguf.py). Each prints counts and exits 1 on any mismatch.

    PYTHONPATH=<llama.cpp b12360>/gguf-py python gguf/check_chat_gguf.py tensors OLD.gguf NEW.gguf
    python gguf/check_chat_gguf.py render-offline            # new template (transformers) == build_prompt(), 2,980 prompts
    python gguf/check_chat_gguf.py render-server --url URL   # llama-server /apply-template == build_prompt(), 300 test rows
    python gguf/check_chat_gguf.py chat --url URL [--limit N]  # /v1/chat/completions == sft_grpo_gguf_q8 raw outputs

tensors:        every tensor has the same name, type, shape and bytes; the metadata differs only by the new template
                and general.sampling.temp. (The file hashes differ: a longer header moves the tensor data.)
render-offline: 2,500 data-v1 chats + the 480 canonical sentences, rendered with and without our SYSTEM message and
                with enable_thinking unset / false / true; each must equal build_prompt() (original template, SYSTEM,
                thinking off).
render-server:  the same on llama.cpp's own Jinja engine, through the chat request path (server defaults included).
chat:           user message only, greedy, no penalties: message.content must equal the raw /completion text saved
                by evaluate.py. A row that differs is asked again through raw /completion on the same server, to
                tell a template or chat-parser problem from a model or server change.
Standard library + transformers (render, chat) + gguf-py (tensors only).
"""
import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from evaluate import DEFAULT_TOKENIZER, OUTPUTS, TEST, build_prompt, read_jsonl  # noqa: E402
from llama_client import GREEDY, MAX_NEW_TOKENS, check_server, complete, http_json  # noqa: E402
from make_dataset import SYSTEM, all_combos  # noqa: E402
from request_parser import canonical_text  # noqa: E402

TEMPLATE = Path(__file__).with_name("chat_template.jinja").read_text(encoding="utf-8")
REFERENCE = "sft_grpo_gguf_q8"
CHANGED = {"tokenizer.chat_template"}
ADDED = {"general.sampling.temp": 0.0}


def tensors(old_path, new_path):
    import numpy as np
    from gguf import GGUFReader
    old, new = GGUFReader(old_path), GGUFReader(new_path)

    # metadata: GGUF.* are the reader's own fields (kv_count grows by one with the added key)
    fields = lambda r: {k: f for k, f in r.fields.items() if not k.startswith("GGUF.")}  # noqa: E731
    a, b = fields(old), fields(new)
    differ = {k for k in a.keys() & b.keys() if (a[k].types, a[k].contents()) != (b[k].types, b[k].contents())}
    added = {k: b[k].contents() for k in b.keys() - a.keys()}
    removed = a.keys() - b.keys()
    print(f"metadata: {len(a.keys() & b.keys()) - len(differ)} keys equal, changed {sorted(differ)}, "
          f"added {added}, removed {sorted(removed)}")
    ok = differ == CHANGED and added == ADDED and not removed and b["tokenizer.chat_template"].contents() == TEMPLATE

    same, total = 0, 0
    if len(old.tensors) != len(new.tensors):
        print(f"tensor count {len(old.tensors)} != {len(new.tensors)}")
        ok = False
    for x, y in zip(old.tensors, new.tensors):
        head = lambda t: (t.name, t.tensor_type, t.shape.tolist(), t.n_bytes)  # noqa: E731
        digest = lambda t: hashlib.sha256(np.ascontiguousarray(t.data).reshape(-1).view(np.uint8)).digest()  # noqa: E731
        if head(x) == head(y) and x.data.nbytes == x.n_bytes and digest(x) == digest(y):
            same += 1
            total += x.n_bytes
        else:
            print(f"  differs: {x.name} vs {y.name}")
    ok = ok and same == len(old.tensors)
    print(f"tensors: {same} / {len(old.tensors)} identical (name, type, shape, bytes), {total:,} tensor bytes")
    print(f"tensor data starts at byte {old.data_offset:,} (old) and {new.data_offset:,} (new)")
    return ok


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER, local_files_only=True)


def user_messages():
    """The 2,500 data-v1 user messages and the 480 canonical sentences."""
    texts = [row["messages"][1]["content"] for split in ("train", "val", "test")
             for row in read_jsonl(ROOT / f"data/processed/{split}.jsonl")]
    return texts + [canonical_text(c) for c in all_combos()]


def chats(text):
    """-> {name: messages}: without and with our system message."""
    user = {"role": "user", "content": text}
    return {"no system": [user], "SYSTEM": [{"role": "system", "content": SYSTEM}, user]}


def render_offline():
    tok = load_tokenizer()
    texts = user_messages()
    thinking = {"unset": {}, "false": {"enable_thinking": False}, "true": {"enable_thinking": True}}
    counts, old_counts = {}, {}
    for text in texts:
        expected = build_prompt(tok, chats(text)["SYSTEM"])
        for name, messages in chats(text).items():
            for t, kwargs in thinking.items():
                got = tok.apply_chat_template(messages, chat_template=TEMPLATE, add_generation_prompt=True,
                                              tokenize=False, **kwargs)
                counts[name, t] = counts.get((name, t), 0) + (got == expected)
                old = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kwargs)
                old_counts[name, t] = old_counts.get((name, t), 0) + (old == expected)
    print(f"{len(texts)} prompts; equal to build_prompt(), new template vs original template:")
    for key in counts:
        print(f"  {key[0]:<9} enable_thinking {key[1]:<5}  new {counts[key]:>5} / {len(texts)}   "
              f"original {old_counts[key]:>5} / {len(texts)}")
    return all(c == len(texts) for c in counts.values())


def server_info(url):
    props = check_server(url)
    template = props.get("chat_template")
    kind = "new (gguf/chat_template.jinja)" if template == TEMPLATE else "NOT the new one"
    print(f"llama-server {url}: {props.get('model_path')}, build {props.get('build_info')}, chat template: {kind}")


def first_difference(a, b):
    i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
    return f"at char {i}: got {a[max(i - 40, 0):i + 40]!r}, expected {b[max(i - 40, 0):i + 40]!r}"


def render_server(url):
    url = url.rstrip("/")
    server_info(url)
    tok = load_tokenizer()
    test = read_jsonl(TEST)
    counts, example = {}, {}
    for row in test:
        text = row["messages"][1]["content"]
        expected = build_prompt(tok, chats(text)["SYSTEM"])
        for name, messages in chats(text).items():
            got = http_json(f"{url}/apply-template", {"messages": messages})["prompt"]
            counts[name] = counts.get(name, 0) + (got == expected)
            if got != expected and name not in example:
                example[name] = first_difference(got, expected)
    for name, n in counts.items():
        print(f"  {name:<9} {n:>3} / {len(test)} equal to build_prompt()"
              + (f"; first difference {example[name]}" if name in example else ""))
    return all(n == len(test) for n in counts.values())


def chat(url, limit=None):
    url = url.rstrip("/")
    server_info(url)
    test, ref = read_jsonl(TEST)[:limit], read_jsonl(OUTPUTS / f"{REFERENCE}.jsonl")[:limit]
    tok = None
    identical, problems = 0, []
    for i, (row, r) in enumerate(zip(test, ref)):
        text = row["messages"][1]["content"]
        assert r["request"] == text, f"row {i}: {REFERENCE} has another request"
        out = http_json(f"{url}/v1/chat/completions", {"messages": chats(text)["no system"], "max_tokens": MAX_NEW_TOKENS,
                                                       "cache_prompt": False, **GREEDY})
        choice = out["choices"][0]
        content, reasoning = choice["message"].get("content"), choice["message"].get("reasoning_content")
        finish = "length" if r["truncated"] else "stop"
        ok = content == r["output"] and not reasoning and choice["finish_reason"] == finish
        identical += ok
        tps = out.get("timings", {}).get("predicted_per_second")
        print(f"[{i + 1}/{len(test)}] {'identical' if ok else 'DIFFERS'}"
              + (f"  {tps:.1f} tok/s" if tps else ""), flush=True)
        if not ok:
            tok = tok or load_tokenizer()
            raw = complete(url, build_prompt(tok, chats(text)["SYSTEM"]))[0]
            cause = "chat path (raw /completion still equals the saved text)" if raw == r["output"] else \
                "model or server (raw /completion differs too)"
            problems.append(i)
            print(f"    row {i}: {text!r}: finish {choice['finish_reason']}, reasoning {reasoning!r}, cause: {cause}\n"
                  f"    {first_difference(content or '', r['output'])}")
    print(f"chat content identical to {REFERENCE}: {identical} / {len(test)}; differing rows: {problems}")
    return not problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    p = sub.add_parser("tensors")
    p.add_argument("old", type=Path)
    p.add_argument("new", type=Path)
    sub.add_parser("render-offline")
    for name in ("render-server", "chat"):
        p = sub.add_parser(name)
        p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    ok = {"tensors": lambda: tensors(args.old, args.new), "render-offline": render_offline,
          "render-server": lambda: render_server(args.url), "chat": lambda: chat(args.url, args.limit)}[args.step]()
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
