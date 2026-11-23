"""Sprawdzenie uzupełniania konfiguracji bez dostępu do sieci i modeli."""
import yaml
import pytest

from src.common import ProtocolError, config, digest, write_json
from src.configure_run import initialize


def test_initialize_and_freeze_guard(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    experiment = tmp_path / "experiment.yaml"
    generator = tmp_path / "generator.yaml"
    evaluation = tmp_path / "evaluation.yaml"
    manifest = tmp_path / "data.json"
    rules = tmp_path / "rules.jsonl"
    experiment.write_text(yaml.safe_dump({"run_id": None, "output_directory": "outputs"}), encoding="utf-8")
    generator.write_text(yaml.safe_dump({"model_name": "generator", "model_revision": "a"*40}), encoding="utf-8")
    evaluation.write_text(yaml.safe_dump({"model_name": "judge", "model_revision": "b"*40,
                                         "output_directory": "evaluation"}), encoding="utf-8")
    write_json(manifest, {"repository_commit": "c"*40})
    rules.write_text("synthetic rules", encoding="utf-8")
    result = initialize(experiment, generator, evaluation, manifest, rules)
    assert result["run_id"].startswith("aci-")
    assert result["code_version"].startswith("sha256:")
    assert result["rules_sha256"] == digest(rules.read_bytes())
    assert initialize(experiment, generator, evaluation, manifest, rules) == result
    write_json(tmp_path / "outputs/frozen_plan.json", {})
    with pytest.raises(ProtocolError, match="frozen run"):
        initialize(experiment, generator, evaluation, manifest, rules)
