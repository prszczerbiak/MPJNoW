"""Testy oceny modelowej na atrapach; bez pobierania wag i generowania wyników badania."""
import copy
import json

import pytest

from src.common import ProtocolError, read_json, write_json
from src.evaluate_notes import (LocalJudge, evaluate_facts, evaluate_notes, export_metrics, fact_key,
                                pending_reviews, sample_ids, submit_review, validate_assessment)
from src.aggregate_results import aggregate


@pytest.fixture
def evaluation_config():
    return {"protocol_version": "model-assisted-v2", "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "model_revision": "a"*40, "seed": 27182, "audit_fraction": .1,
            "reviewers": ["student_1", "student_2"]}


@pytest.fixture
def generations(records):
    return [{"run_id": "synthetic", "task_id": f'{r["encounter_id"]}-{v}-{s}',
             "encounter_id": r["encounter_id"], "variant": v, "seed": s,
             "split": "test1", "environment_id": "synthetic-cpu", "status": "ok",
             "response": "The patient has a cough."}
            for r in records for v in "EABC" for s in ([None] if v == "E" else [17, 42, 2026])]


class FakeJudge:
    def __init__(self):
        self.calls = []

    def __call__(self, stage, payload):
        self.calls.append((stage, payload))
        if stage == "facts":
            assert set(payload) == {"transcript"}
            return json.dumps({"facts": [{"fact_id": "F001", "category": "symptoms", "text": "Cough",
                                         "source_span": "I have a cough.", "rationale": "Presenting symptom",
                                         "uncertain": False}], "review_required": False})
        result = {"labels": [{"claim_id": "C001", "label": "P", "source_span": "I have a cough.",
                              "rationale": "Explicit report", "uncertain": False}],
                  "coverage": [{"fact_id": "F001", "covered": True, "claim_ids": ["C001"],
                                "rationale": "Symptom reproduced", "uncertain": False}], "review_required": False}
        if stage == "assessment":
            result.update(response_kind="clinical", claims=[{"claim_id": "C001", "text": "Cough",
                                                              "note_span": "The patient has a cough."}])
        else:
            assert "labels" not in payload and "coverage" not in payload
        return json.dumps(result)


def approve_all(directory):
    for i, path in enumerate(pending_reviews(directory)):
        packet = read_json(path)
        value = copy.deepcopy(packet["proposal"])
        value["review_required"] = False
        for group in ("facts", "labels", "coverage"):
            for item in value.get(group, []):
                item["uncertain"] = False
        submit_review(directory, path.stem, f"student_{1+i%2}", value, "Checked against the transcript")


def test_full_model_assisted_flow(tmp_path, records, generations, evaluation_config):
    judge = FakeJudge()
    env = {"synthetic": True}
    evaluate_facts(records, generations, evaluation_config, judge, tmp_path, env)
    assert len(pending_reviews(tmp_path)) == 1
    with pytest.raises(ProtocolError, match="fact reviews"):
        evaluate_notes(records, generations, evaluation_config, judge, tmp_path, env)
    approve_all(tmp_path)
    evaluate_notes(records, generations, evaluation_config, judge, tmp_path, env)
    assert len(pending_reviews(tmp_path)) == 4
    with pytest.raises(ProtocolError, match="Pending reviews"):
        export_metrics(records, generations, tmp_path)
    for path in pending_reviews(tmp_path):
        packet = read_json(path)
        assert "variant" not in packet and "seed" not in packet and "task_id" not in packet
    approve_all(tmp_path)
    frame = export_metrics(records, generations, tmp_path)
    assert len(frame) == 30
    assert set(frame.evaluation_method) == {"model_assisted_v2"}
    assert set(frame.assessment_origin) == {"automatic", "human_review"}
    assert (frame.H_star == 0).all() and (frame.C == 1).all()
    assert read_json(tmp_path / "summary.json")["human_interrater_agreement"] is None
    aggregate(frame)
    n_calls = len(judge.calls)
    evaluate_notes(records, generations, evaluation_config, judge, tmp_path, env)
    assert len(judge.calls) == n_calls
    assert all("variant" not in p and "seed" not in p for _, p in judge.calls)


def test_disagreement_requires_review(tmp_path, records, generations, evaluation_config):
    judge = FakeJudge()
    evaluate_facts(records, generations, evaluation_config, judge, tmp_path, {})
    approve_all(tmp_path)
    def disagree(stage, payload):
        value = json.loads(judge(stage, payload))
        if stage == "check":
            value["labels"][0]["label"] = "N"
            value["coverage"][0].update(covered=False, claim_ids=[])
        return json.dumps(value)
    evaluate_notes(records, generations, evaluation_config, disagree, tmp_path, {})
    assert len(pending_reviews(tmp_path)) == 30
    assert all("inconsistent_checks" in read_json(p)["reasons"] for p in pending_reviews(tmp_path))


def test_invented_source_evidence_is_not_accepted(tmp_path, records, generations, evaluation_config):
    judge = FakeJudge()
    evaluate_facts(records, generations, evaluation_config, judge, tmp_path, {})
    approve_all(tmp_path)
    def invalid(stage, payload):
        value = json.loads(judge(stage, payload))
        value["labels"][0]["source_span"] = "Invented quote"
        return json.dumps(value)
    evaluate_notes(records, generations, evaluation_config, invalid, tmp_path, {})
    assert len(pending_reviews(tmp_path)) == 30
    assert not list((tmp_path / "decisions").glob("*.json"))


def test_invalid_json_is_cached_and_sent_to_review(tmp_path, records, generations, evaluation_config):
    calls = []
    def invalid(stage, payload):
        calls.append(stage)
        return "Not JSON"
    evaluate_facts(records, generations, evaluation_config, invalid, tmp_path, {})
    evaluate_facts(records, generations, evaluation_config, invalid, tmp_path, {})
    assert len(calls) == 3 and len(pending_reviews(tmp_path)) == 3
    assert all(read_json(p)["proposal"] is None for p in pending_reviews(tmp_path))


def test_sampling_is_stable_stratified_and_nonempty():
    rows = [{"id": f"{v}-{i}", "variant": v} for v in "EABC" for i in range(11)]
    a = sample_ids(rows, .1, 27182, "id", "variant")
    b = sample_ids(list(reversed(rows)), .1, 27182, "id", "variant")
    assert a == b and len(a) == 8
    with pytest.raises(ProtocolError):
        sample_ids(rows, 0, 1, "id")


def test_provenance_change_blocks_resume(tmp_path, records, generations, evaluation_config):
    evaluate_facts(records, generations, evaluation_config, FakeJudge(), tmp_path, {"gpu": "one"})
    with pytest.raises(ProtocolError, match="frozen"):
        evaluate_facts(records, generations, evaluation_config, FakeJudge(), tmp_path, {"gpu": "two"})
    changed = copy.deepcopy(generations)
    changed[0]["response"] = "Changed"
    with pytest.raises(ProtocolError, match="inputs changed"):
        export_metrics(records, changed, tmp_path)


def test_technical_failures_are_retained(tmp_path, records, generations, evaluation_config):
    generations[0]["status"] = "technical_failure"
    judge = FakeJudge()
    evaluate_facts(records, generations, evaluation_config, judge, tmp_path, {})
    approve_all(tmp_path)
    evaluate_notes(records, generations, evaluation_config, judge, tmp_path, {})
    approve_all(tmp_path)
    frame = export_metrics(records, generations, tmp_path)
    assert len(frame) == len(generations)
    assert frame.loc[frame.status == "technical_failure", "H_star"].isna().all()


def test_no_model_generator_self_evaluation(tmp_path, records, generations, evaluation_config):
    evaluation_config["model_name"] = "Qwen/Qwen2.5-3B-Instruct"
    with pytest.raises(ProtocolError, match="note generator"):
        evaluate_facts(records, generations, evaluation_config, FakeJudge(), tmp_path, {})


def test_mixed_assessment_sources_rejected(tmp_path, records, generations, evaluation_config):
    judge = FakeJudge()
    evaluate_facts(records, generations, evaluation_config, judge, tmp_path, {})
    approve_all(tmp_path)
    evaluate_notes(records, generations, evaluation_config, judge, tmp_path, {})
    approve_all(tmp_path)
    frame = export_metrics(records, generations, tmp_path)
    frame.loc[0, "evaluation_method"] = "human"
    with pytest.raises(ProtocolError, match="human and model"):
        aggregate(frame)


def test_empty_responses_have_no_judge_calls_and_hstar_one(tmp_path, records, generations, evaluation_config):
    for row in generations:
        row["response"] = ""
    judge = FakeJudge()
    evaluate_facts(records, generations, evaluation_config, judge, tmp_path, {})
    approve_all(tmp_path)
    evaluate_notes(records, generations, evaluation_config, judge, tmp_path, {})
    approve_all(tmp_path)
    frame = export_metrics(records, generations, tmp_path)
    assert len(judge.calls) == len(records)
    assert (frame.H_star == 1).all() and frame.H.isna().all() and (frame.C == 0).all()


def test_context_limit_blocks_before_model_call():
    import torch
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return torch.zeros((1, 9), dtype=torch.long)
    judge = LocalJudge.__new__(LocalJudge)
    judge.cfg = {"max_new_tokens": 4, "context_budget": 12}
    judge.tokenizer = Tokenizer()
    with pytest.raises(ProtocolError, match="no source truncation"):
        judge("facts", {"transcript": "Synthetic source"})


def test_fact_changes_after_review_are_detected(tmp_path, records, generations, evaluation_config):
    judge = FakeJudge()
    evaluate_facts(records, generations, evaluation_config, judge, tmp_path, {})
    approve_all(tmp_path)
    evaluate_notes(records, generations, evaluation_config, judge, tmp_path, {})
    approve_all(tmp_path)
    path = tmp_path / "facts" / f'{fact_key(records[0]["encounter_id"])}.json'
    modified = read_json(path)
    modified["value"]["facts"][0]["text"] = "Modified fact"
    write_json(path, modified)
    with pytest.raises(ProtocolError, match="provenance"):
        export_metrics(records, generations, tmp_path)


def test_audit_report_counts_corrections(tmp_path, records, generations, evaluation_config):
    judge = FakeJudge()
    evaluate_facts(records, generations, evaluation_config, judge, tmp_path, {})
    approve_all(tmp_path)
    evaluate_notes(records, generations, evaluation_config, judge, tmp_path, {})
    path = pending_reviews(tmp_path)[0]
    packet = read_json(path)
    changed = copy.deepcopy(packet["proposal"])
    changed["labels"][0]["label"] = "N"
    changed["coverage"][0].update(covered=False, claim_ids=[])
    submit_review(tmp_path, path.stem, "student_1", changed, "Synthetic correction for a test")
    approve_all(tmp_path)
    export_metrics(records, generations, tmp_path)
    report = read_json(tmp_path / "summary.json")
    assert report["random_audit"]["note"] == {"reviewed": 4, "corrected": 1}


def test_review_rejects_undocumented_or_unknown_reviewer(tmp_path, records, generations, evaluation_config):
    evaluate_facts(records, generations, evaluation_config, FakeJudge(), tmp_path, {})
    path = pending_reviews(tmp_path)[0]
    value = read_json(path)["proposal"]
    with pytest.raises(ProtocolError, match="reviewer"):
        submit_review(tmp_path, path.stem, "model", value, "Approved")
    with pytest.raises(ProtocolError, match="rationale"):
        submit_review(tmp_path, path.stem, "student_1", value, "")
