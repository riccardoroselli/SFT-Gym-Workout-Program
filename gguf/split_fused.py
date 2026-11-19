"""Copy a qwen35 GGUF with the fused linear-attention tensors split back into the older layout.

    PYTHONPATH=<llama.cpp b12360>/gguf-py python gguf/split_fused.py IN.gguf OUT.gguf

llama.cpp's converter (since b12343, PR #32102) writes [attn_qkv | attn_gate] as one attn_qkv_gate tensor and
[ssm_beta | ssm_alpha] as one ssm_beta_alpha tensor: torch.cat(..., dim=0), i.e. the rows of the first part
followed by the rows of the second. Loaders older than b12343 (e.g. LM Studio's engine 2.73.0 = b12270) only
know the separate tensors; newer loaders read both. Every quantized block covers part of a single row, so cutting
between rows moves bytes without changing any. Metadata and all other tensors are copied unchanged (same gguf-py
code path as gguf/make_chat_gguf.py). Check with: python gguf/check_chat_gguf.py split-check IN OUT REFERENCE.
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import gguf
from gguf.scripts.gguf_new_metadata import copy_with_new_metadata, get_field_data

# fused suffix -> (first part, second part, metadata key giving the second part's row count)
FUSED = {
    "attn_qkv_gate.weight": ("attn_qkv.weight", "attn_gate.weight", "ssm.inner_size"),
    "ssm_beta_alpha.weight": ("ssm_beta.weight", "ssm_alpha.weight", "ssm.time_step_rank"),
}


def split_tensors(reader):
    """-> the reader's tensors with each fused tensor replaced by its two row ranges (same type, same bytes)."""
    arch = get_field_data(reader, gguf.Keys.General.ARCHITECTURE)
    out = []
    for t in reader.tensors:
        block, _, suffix = t.name.partition(".")[2].partition(".")
        if not t.name.startswith("blk.") or suffix not in FUSED:
            out.append(t)
            continue
        first, second, key = FUSED[suffix]
        rows = len(t.data) - get_field_data(reader, f"{arch}.{key}")  # data: one numpy row per tensor row
        assert t.data.ndim == 2 and 0 < rows < len(t.data), (t.name, t.data.shape)
        for name, data in ((first, t.data[:rows]), (second, t.data[rows:])):
            out.append(SimpleNamespace(name=f"blk.{block}.{name}", data=data, n_bytes=data.nbytes,
                                       tensor_type=t.tensor_type))
    return out


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
    tensors = split_tensors(reader)
    copy_with_new_metadata(SimpleNamespace(fields=reader.fields, tensors=tensors, endianess=reader.endianess),
                           writer, {}, [])
    print(f"wrote {args.output}: {len(reader.tensors)} -> {len(tensors)} tensors")


if __name__ == "__main__":
    main()
