"""Test integracyjny na danych syntetycznych, bez pobierania wag i wywołań API."""
import csv
import numpy as np
import pytest
from src.prepare_data import load_csv, validate_records
from src.retrieve_rules import rank_rules, windows
from src.generate_notes import plan_tasks, execute
from src.prepare_annotation import prepare_facts, prepare_notes
from src.annotate import submit_facts, freeze_facts, freeze_claims, submit_rating, adjudicate, export_agreement
from src.compute_metrics import compute
from src.aggregate_results import aggregate, paired_tests
from src.common import ProtocolError, read_json


def test_synthetic_end_to_end(tmp_path, rules, tokenizer, configs):
    csv_path = tmp_path / "source.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "encounter_id", "dialogue", "note"])
        for i in range(3):
            writer.writerow(["synthetic", f"synthetic-{i}", f"[doctor] Visit {i}.\n[patient] I have a cough.", f"Unused reference {i}"])
    rows = load_csv(csv_path, "test1")
    assert not validate_records(rows, {"test1": 3})["duplicate_texts"]
    retrieved = []
    for row in rows:
        chunks = windows(row["transcript"], tokenizer)
        retrieved.extend(rank_rules(row["encounter_id"], rules, [[1, 0]]*60, chunks, [[1, 0]]*len(chunks)))
    plan = plan_tasks(rows, rules, retrieved, tokenizer, *configs)
    generated = execute(plan, lambda prompt, seed: ("History\nThe patient has a cough.", 8),
                        {"environment": "synthetic_cpu_only"}, tmp_path / "outputs")
    directory = tmp_path / "annotations"
    prepare_facts(rows, directory)
    with pytest.raises(ProtocolError, match="Freeze"):
        prepare_notes(generated, directory)
    for row in rows:
        fact = {"fact_id": "f1", "text": "Cough", "category": "symptoms",
                "source_span": "I have a cough.", "rationale": "Presenting symptom"}
        for aid in ["annotator_1", "annotator_2"]:
            submit_facts(directory, {"encounter_id": row["encounter_id"], "annotator_id": aid, "facts": [fact]})
        freeze_facts(directory, {"encounter_id": row["encounter_id"], "adjudicator_id": "annotator_3",
                                 "rationale": "Independent lists agree", "facts": [fact]})
    key = prepare_notes(generated, directory)
    for entry in key:
        bid = entry["blind_id"]
        packet = read_json(directory / "segmentation" / f"{bid}.json")
        assert "variant" not in packet and "seed" not in packet
        # E kopiuje tekst źródłowy, a atrapa generatora zwraca stałą notatkę syntetyczną.
        span = "I have a cough." if entry["variant"] == "E" else "The patient has a cough."
        freeze_claims(directory, {"blind_id": bid, "reviewed_by": ["annotator_1", "annotator_2"],
                                  "response_kind": "clinical", "claims": [{"claim_id": "c1", "text": "Cough", "note_span": span}]})
        rating = {"blind_id": bid,
                  "labels": [{"claim_id": "c1", "label": "P", "source_span": "I have a cough.", "rationale": "Explicit report"}],
                  "coverage": [{"fact_id": "f1", "covered": True, "claim_ids": ["c1"], "rationale": "Cough reproduced"}]}
        for aid in ["annotator_1", "annotator_2"]:
            submit_rating(directory, {**rating, "annotator_id": aid})
        adjudicate(directory, {**rating, "adjudicator_id": "annotator_3", "rationale": "Independent ratings agree"})
    frame = compute(generated, directory)
    means, macro, _ = aggregate(frame, plan["tasks"])
    assert len(frame) == 30 and len(means) == 12
    assert (macro.H_star == 0).all() and (macro.C == 1).all()
    export_agreement(directory, tmp_path / "agreement.json")
    assert read_json(tmp_path / "agreement.json")["raw_agreement"] == 1
    # Jednakowa kompletność daje zerową wariancję różnic i blokuje test t.
    with pytest.raises(ProtocolError, match="zero variance"):
        paired_tests(means, n_resamples=99, bootstrap_resamples=100)


def test_dependency_integration():
    import torch
    import accelerate
    import sentence_transformers
    import transformers
    from transformers import Qwen2Config, Qwen2ForCausalLM
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                                        num_hidden_layers=1, num_attention_heads=2,
                                        num_key_value_heads=2, max_position_embeddings=32))
    model.eval()
    with torch.inference_mode():
        output = model(input_ids=torch.tensor([[1, 2, 3]]))
    assert output.logits.shape == (1, 3, 32)
