"""The Hub model cards in cards/ (uploaded by hand as each repo's README.md): the agreed headers,
the GGUF card's files equal to what serve.py downloads, and links that point only to things that exist."""
import re
from pathlib import Path

import pytest
from huggingface_hub import ModelCard

import serve

ROOT = Path(__file__).resolve().parent.parent
CARDS = ROOT / "cards"
QWEN, ME = "Qwen/Qwen3.5-2B", "riccardoroselli/qwen3.5-2b-gym"
# repo -> (base_model, base_model_relation, library_name, new_version)
HEADERS = {
    "qwen3.5-2b-gym-sft": (QWEN, "finetune", "transformers", f"{ME}-grpo"),
    "qwen3.5-2b-gym-sft-lora": (QWEN, "adapter", "peft", f"{ME}-grpo-lora"),
    "qwen3.5-2b-gym-grpo": (f"{ME}-sft", "finetune", "transformers", None),  # GRPO started from the SFT weights
    "qwen3.5-2b-gym-grpo-lora": (QWEN, "adapter", "peft", None),  # adapter_config.json: Qwen/Qwen3.5-2B
    "qwen3.5-2b-gym-grpo-GGUF": (f"{ME}-grpo", "quantized", None, None),
}
SYSTEM_LINE = "**The GymLeader system, not the model:**"


def text(repo):
    return (CARDS / f"{repo}.md").read_text()


def test_one_card_per_repo():
    assert sorted(p.stem for p in CARDS.glob("*.md")) == sorted(HEADERS)


@pytest.mark.parametrize("repo", HEADERS)
def test_header(repo):
    data = ModelCard.load(CARDS / f"{repo}.md").data.to_dict()
    base, relation, library, new_version = HEADERS[repo]
    assert data["license"] == "apache-2.0" and data["language"] == ["en"] and data["pipeline_tag"] == "text-generation"
    assert (data["base_model"], data["base_model_relation"]) == (base, relation)
    assert data.get("library_name") == library and data.get("new_version") == new_version


@pytest.mark.parametrize("repo", HEADERS)
def test_links_point_to_things_that_exist(repo):
    card = text(repo)
    hf = set(re.findall(r"huggingface\.co/([\w.-]+/[\w.-]+)", card))
    assert hf <= {QWEN, *(f"riccardoroselli/{r}" for r in HEADERS)}, hf
    for path in re.findall(r"SFT-Gym-Workout-Program/blob/main/([\w./-]+)", card):
        assert (ROOT / path).is_file(), path
    assert "/tree/v1.0" not in card and "PENDING" not in card


def test_same_system_line_everywhere():
    lines = {next(l for l in text(r).splitlines() if l.startswith(SYSTEM_LINE)) for r in HEADERS}
    assert len(lines) == 1
    line = lines.pop()
    assert all(s in line for s in ("0.82 all_correct", "MAX_RETRIES=3", "not tuned on the test set", "Metal", "0.55"))


def test_gguf_card_lists_the_files_serve_py_downloads():
    card = text("qwen3.5-2b-gym-grpo-GGUF")
    d = serve.DEFAULTS
    q8_row = next(l for l in card.splitlines() if l.startswith(f"| `{d['MODEL_FILE']}`"))
    assert d["MODEL_SHA256"] in q8_row and "2,076,675,008 bytes" in q8_row
    assert card.count(d["MODEL_REVISION"]) >= 2  # the files table and the download command
    assert "ade25302cea60cf78432b0f92b491e93b7cdcfabed29e70a18d6cd4d8cceb20c" in card  # BF16, as in eval/gguf_notes.md "sha256"
