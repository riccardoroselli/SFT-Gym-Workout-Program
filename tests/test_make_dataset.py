"""error_answer() is the text the API answers impossible requests with: it must be data-v1's, byte for byte."""
from evaluate import ROOT, read_jsonl
from make_dataset import error_answer


def test_error_answer_equals_every_impossible_data_v1_answer():
    rows = [r for split in ("train", "val", "test") for r in read_jsonl(ROOT / f"data/processed/{split}.jsonl")
            if not r["meta"]["feasible"]]
    assert len(rows) == 125 + 13 + 16
    for r in rows:
        assert error_answer(r["meta"]) == r["messages"][2]["content"], r["meta"]
