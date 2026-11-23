import numpy as np
import pytest
from src.build_index import validate_rules
from src.common import ProtocolError
from src.retrieve_rules import rank_rules, windows


@pytest.mark.parametrize("count,expected", [(1, 1), (128, 1), (129, 2), (224, 2), (225, 3)])
def test_windows(tokenizer, count, expected):
    chunks = windows(" ".join(f"w{i}" for i in range(count)), tokenizer)
    assert len(chunks) == expected
    assert chunks[-1]["end_token"] == count
    assert all(c["token_count"] <= 128 for c in chunks)
    if len(chunks) > 1:
        assert chunks[0]["end_token"] - chunks[1]["start_token"] == 32


def test_invalid_windows(tokenizer):
    with pytest.raises(ProtocolError):
        windows("", tokenizer)
    with pytest.raises(ProtocolError):
        windows("x", tokenizer, size=32, overlap=32)


def test_max_cosine_and_ties(tokenizer):
    rules = [{"rule_id": r} for r in ["G04", "G02", "G03", "G01", "G05"]]
    chunks = windows("one two three", tokenizer, size=2, overlap=1)
    ranked = rank_rules("x", rules, [[1, 0]]*4 + [[0, 1]], chunks, [[1, 0], [0, -1]])
    assert [r["rule_id"] for r in ranked[:4]] == ["G01", "G02", "G03", "G04"]
    assert sum(r["selected"] for r in ranked) == 4
    assert ranked[0]["selected_window_id"] == 0
    assert ranked[0]["similarity_score"] == 1


def test_rules_schema_and_fixed_text(rules, tokenizer):
    counts = validate_rules(rules, tokenizer, tokenizer)
    assert len(counts) == 60
    rules[0]["text"] = "Altered"
    with pytest.raises(ProtocolError, match="Fixed"):
        validate_rules(rules)


def test_actual_window_token_ids_are_forwarded_without_retokenization(tokenizer):
    import torch
    from src.retrieve_rules import encode_windows
    class Encoder:
        device = "cpu"
        max_seq_length = 256
        def __init__(self):
            self.seen = []
            self.tokenizer = self
        def eval(self):
            return self
        def build_inputs_with_special_tokens(self, ids):
            return [101] + ids + [102]
        def __call__(self, features):
            self.seen.append(features["input_ids"].tolist()[0])
            return {"sentence_embedding": torch.tensor([[3., 4.]])}
    encoder = Encoder()
    chunks = windows(" ".join("word" for _ in range(129)), tokenizer)
    vectors = encode_windows(encoder, chunks)
    assert encoder.seen[0] == [101, *range(128), 102]
    assert encoder.seen[1] == [101, *range(96, 129), 102]
    assert np.allclose(vectors, [[.6, .8], [.6, .8]])
