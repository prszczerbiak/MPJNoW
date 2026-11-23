"""Testy profilu 3B i kwantyzacji bez ładowania wag ani używania GPU."""
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import transformers

from src.common import ProtocolError, config
from src.generate_notes import LocalGenerator, MODEL, execute, plan_tasks, validate_config
from src.evaluate_notes import LocalJudge
from src.model_loading import EVALUATOR_MODEL, loading_kwargs, validate_loading


def test_generator_identity_and_4bit_profile(configs):
    gen, exp = configs
    assert MODEL == gen["model_name"] == gen["tokenizer_name"] == "Qwen/Qwen2.5-3B-Instruct"
    assert gen["quantization"] == "4bit" and gen["quantization_type"] == "nf4"
    assert gen["double_quantization"] is True
    validate_config(gen, exp)
    gen["model_name"] = "Qwen/Qwen2.5-7B-Instruct"
    with pytest.raises(ProtocolError, match="3B"):
        validate_config(gen, exp)


def test_no_silent_cpu_fallback(configs, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ProtocolError, match="CUDA-enabled"):
        loading_kwargs(configs[0])


def test_nf4_arguments_and_dtype(configs, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    kwargs = loading_kwargs(configs[0])
    quantization = kwargs["quantization_config"]
    assert quantization.load_in_4bit and not quantization.load_in_8bit
    assert quantization.bnb_4bit_quant_type == "nf4"
    assert quantization.bnb_4bit_use_double_quant
    assert quantization.bnb_4bit_compute_dtype == kwargs["torch_dtype"] == torch.bfloat16
    assert kwargs["device_map"] == "cuda:0"


def test_unsupported_dtype_or_quantization(configs, monkeypatch):
    gen = configs[0]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(ProtocolError, match="bfloat16"):
        loading_kwargs(gen)
    gen["quantization_type"] = "invalid"
    with pytest.raises(ProtocolError, match="quantization type"):
        validate_loading(gen)


def test_loaders_use_distinct_models_and_quantization(configs, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    tokenizer_loader = Mock()
    model_loader = Mock()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", model_loader)
    LocalGenerator(configs[0])
    evaluator = config(Path(__file__).parents[1] / "configs/evaluation.yaml")
    evaluator["model_revision"] = "b"*40
    LocalJudge(evaluator)
    assert [call.args[0] for call in model_loader.call_args_list] == [MODEL, EVALUATOR_MODEL]
    assert EVALUATOR_MODEL == "Qwen/Qwen2.5-1.5B-Instruct"
    assert all(call.kwargs["quantization_config"].load_in_4bit for call in model_loader.call_args_list)
    assert model_loader.call_args_list[0].kwargs["revision"] == configs[0]["model_revision"]
    assert model_loader.call_args_list[1].kwargs["revision"] == "b"*40


def test_generation_records_identify_3b(tmp_path, records, rules, retrieval, tokenizer, configs):
    plan = plan_tasks(records, rules, retrieval, tokenizer, *configs)
    rows = execute(plan, lambda p, s: ("Synthetic response", 2), {"synthetic": True}, tmp_path)
    assert {r["model"] for r in rows if r["variant"] != "E"} == {MODEL}
    assert {r["model"] for r in rows if r["variant"] == "E"} == {"extractive"}
