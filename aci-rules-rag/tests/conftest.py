import re
from pathlib import Path
import pytest
from src.common import config, read_jsonl


class FakeTokenizer:
    """Tokenizer dzielący po białych znakach, używany wyłącznie w testach."""
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def __call__(self, text, **kwargs):
        matches = list(re.finditer(r"\S+", text))
        return {"input_ids": list(range(len(matches))), "offset_mapping": [(m.start(), m.end()) for m in matches]}

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(m["role"] + "\n" + m["content"] for m in messages) + "\nassistant\n"


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def rules():
    return read_jsonl(Path(__file__).parents[1] / "rules/rules.jsonl")


@pytest.fixture
def configs():
    root = Path(__file__).parents[1]
    gen = config(root / "configs/generator.yaml")
    exp = config(root / "configs/experiment.yaml")
    gen["model_revision"] = "a" * 40
    exp.update(run_id="synthetic-test", code_version="synthetic-test")
    return gen, exp


@pytest.fixture
def records():
    return [{"encounter_id": f"synthetic-{i}", "split": "test1", "source_type": "synthetic",
             "transcript": "[doctor] What brings you here?\n[patient] I have a cough."} for i in range(3)]


@pytest.fixture
def retrieval(records, rules):
    return [{"encounter_id": r["encounter_id"], "rule_id": g["rule_id"], "rank": i+1,
             "selected": i < 4, "similarity_score": 1-i/100} for r in records for i, g in enumerate(rules)]
