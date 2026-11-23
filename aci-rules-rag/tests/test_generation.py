import re
import pytest
from src.common import ProtocolError, read_json
from src.generate_notes import execute, extractive, plan_tasks, validate_config


def test_same_prompts_outside_rules(records, rules, retrieval, tokenizer, configs):
    gen, exp = configs
    plan = plan_tasks(records, rules, retrieval, tokenizer, gen, exp)
    tasks = [t for t in plan["tasks"] if t["encounter_id"] == records[0]["encounter_id"] and t["variant"] != "E"]
    stripped = {re.sub(r"<summarisation_rules>.*?</summarisation_rules>", "", t["prompt"], flags=re.S) for t in tasks}
    assert len(stripped) == 1
    assert {t["seed"] for t in tasks} == {17, 42, 2026}
    assert all(len(t["retrieved_rules"]) == (0 if t["variant"] == "A" else 4) for t in tasks)
    assert plan == plan_tasks(records, rules, retrieval, tokenizer, gen, exp)
    assert len(plan["tasks"]) == 30


def test_reference_never_enters_prompt(records, rules, retrieval, tokenizer, configs):
    records[0]["reference_note"] = "SECRET_REFERENCE_SENTINEL"
    plan = plan_tasks(records, rules, retrieval, tokenizer, *configs)
    assert not any("SECRET_REFERENCE_SENTINEL" in t["prompt"] for t in plan["tasks"])


def test_joint_budget_exclusion(records, rules, retrieval, tokenizer, configs):
    records[0]["transcript"] += " word" * 7200
    plan = plan_tasks(records, rules, retrieval, tokenizer, *configs)
    assert plan["blocked"]
    assert len(plan["exclusions"]) == 1
    assert all(t["encounter_id"] != records[0]["encounter_id"] for t in plan["tasks"])


def test_extract_order_and_limits():
    text = "\n".join(f"[patient] turn{i}" for i in range(15))
    response, meta = extractive(text)
    assert meta["utterance_indices"] == [0, 1, *range(5, 15)]
    assert response.startswith("[patient] turn0\n[patient] turn1")
    short = "[doctor] Hello\n[patient] Hello"
    assert extractive(short)[0] == short
    response, meta = extractive("[patient] " + "word "*700 + "\n[doctor] done")
    assert len(response.split()) == 500
    assert meta["single_utterance_truncated"]
    assert meta["utterance_indices"] == [0]


def test_infrastructure_retry_and_content_no_retry(tmp_path, records, rules, retrieval, tokenizer, configs):
    plan = plan_tasks(records, rules, retrieval, tokenizer, *configs)
    calls = []
    def failing(prompt, seed):
        calls.append((prompt, seed))
        raise OSError("synthetic infrastructure failure")
    rows = execute(plan, failing, {"test": True}, tmp_path)
    assert len(calls) == 27*3
    assert sum(r["status"] == "technical_failure" for r in rows) == 27
    execute(plan, failing, {"test": True}, tmp_path)
    assert len(calls) == 81
    other = tmp_path / "refusal"
    calls.clear()
    def refusal(prompt, seed):
        calls.append(seed)
        return "I cannot answer.", 4
    execute(plan, refusal, {"test": True}, other)
    assert len(calls) == 27


def test_environment_mixing_blocked(tmp_path, records, rules, retrieval, tokenizer, configs):
    plan = plan_tasks(records, rules, retrieval, tokenizer, *configs)
    execute(plan, lambda p, s: ("", 0), {"environment": 1}, tmp_path)
    with pytest.raises(ProtocolError, match="frozen"):
        execute(plan, lambda p, s: ("", 0), {"environment": 2}, tmp_path)


def test_config_requires_instruction_model(configs):
    gen, exp = configs
    validate_config(gen, exp)
    gen["model_name"] = "Qwen/Qwen2.5-3B"
    with pytest.raises(ProtocolError):
        validate_config(gen, exp)


def test_exactly_five_percent_exclusion_allowed(records, rules, tokenizer, configs):
    from copy import deepcopy
    records = [{**records[0], "encounter_id": f"case-{i}"} for i in range(20)]
    retrieved = [{"encounter_id": r["encounter_id"], "rule_id": rule["rule_id"], "rank": j+1,
                  "selected": j < 4, "similarity_score": 1-j/100}
                 for r in records for j, rule in enumerate(rules)]
    records[0]["transcript"] += " x"*7200
    plan = plan_tasks(records, rules, retrieved, tokenizer, *configs)
    assert plan["exclusion_fraction"] == .05 and not plan["blocked"]
    records[1]["transcript"] += " x"*7200
    assert plan_tasks(records, rules, retrieved, tokenizer, *configs)["blocked"]


def test_unsorted_retrieval_rejected(records, rules, retrieval, tokenizer, configs):
    retrieval[0]["similarity_score"] = -1
    with pytest.raises(ProtocolError, match="order"):
        plan_tasks(records, rules, retrieval, tokenizer, *configs)


def test_budget_can_be_exceeded_by_rules_only(records, rules, retrieval, tokenizer, configs):
    gen, exp = configs
    initial = plan_tasks(records, rules, retrieval, tokenizer, gen, exp)
    a_count = next(t["input_tokens"] for t in initial["tasks"] if t["variant"] == "A")
    # A mieści się dokładnie w limicie, ale przekroczenie w B/C wyklucza całą konsultację.
    records[0]["transcript"] += " x" * (gen["max_prompt_tokens"]-a_count)
    plan = plan_tasks(records, rules, retrieval, tokenizer, gen, exp)
    assert plan["exclusions"][0]["input_tokens"]["A"] == gen["max_prompt_tokens"]
    assert plan["exclusions"][0]["input_tokens"]["B"] > gen["max_prompt_tokens"]


def test_partial_infrastructure_failure_reuses_input_and_seed(tmp_path, records, rules, retrieval, tokenizer, configs):
    plan = plan_tasks(records, rules, retrieval, tokenizer, *configs)
    task = next(t for t in plan["tasks"] if t["variant"] == "A")
    plan["tasks"] = [task]
    seen = []
    def one_failure(prompt, seed):
        seen.append((prompt, seed))
        if len(seen) == 1:
            raise RuntimeError("Transient synthetic failure")
        return "", 0
    rows = execute(plan, one_failure, {"synthetic": True}, tmp_path)
    assert len(seen) == 2 and seen[0] == seen[1]
    assert rows[0]["status"] == "ok" and rows[0]["response"] == ""
