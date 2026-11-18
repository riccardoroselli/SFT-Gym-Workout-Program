"""Copy a GGUF with gguf/chat_template.jinja and general.sampling.temp = 0.0. The tensors are copied byte for byte.

    PYTHONPATH=<llama.cpp b12360>/gguf-py python gguf/make_chat_gguf.py IN.gguf OUT.gguf

The template is Qwen3.5's own with two edits: our training SYSTEM prompt when the chat has none, and the empty
<think> block always (llama-server sends enable_thinking=true by default). general.sampling.temp makes llama.cpp
greedy unless the user sets --temp or a request sends a temperature. Same code path as gguf-py's
scripts/gguf_new_metadata.py (copy_with_new_metadata), with one extra key that script has no flag for.
"""
import argparse
import sys
from pathlib import Path

import gguf
from gguf.scripts.gguf_new_metadata import MetadataDetails, copy_with_new_metadata, get_field_data

TEMPLATE = Path(__file__).with_name("chat_template.jinja")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        sys.exit(f"{args.output} exists: not overwriting it")

    reader = gguf.GGUFReader(args.input, "r")
    writer = gguf.GGUFWriter(args.output, arch=get_field_data(reader, gguf.Keys.General.ARCHITECTURE),
                             endianess=reader.endianess)
    alignment = get_field_data(reader, gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment
    new_metadata = {
        gguf.Keys.Tokenizer.CHAT_TEMPLATE: MetadataDetails(gguf.GGUFValueType.STRING, TEMPLATE.read_text(encoding="utf-8")),
        gguf.Keys.General.SAMPLING_TEMP: MetadataDetails(gguf.GGUFValueType.FLOAT32, 0.0),
    }
    copy_with_new_metadata(reader, writer, new_metadata, [])
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
