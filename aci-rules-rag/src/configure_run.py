"""Uzupełnia identyfikator przebiegu i sumy źródeł bez uruchamiania modeli."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import yaml

from .common import ProtocolError, atomic_write, config, digest, fingerprint, read_json


def initialize(experiment_path, generator_path, data_manifest, rules_path):
    experiment = config(experiment_path)
    generator = config(generator_path)
    manifest = read_json(data_manifest)
    if (Path(experiment["output_directory"]) / "frozen_plan.json").exists():
        raise ProtocolError("Existing frozen run; use a separate configuration and output directory")
    source_hash = fingerprint({p.name: digest(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))})
    experiment["run_id"] = experiment.get("run_id") or datetime.now(timezone.utc).strftime("aci-%Y%m%dT%H%M%SZ")
    experiment["code_version"] = "sha256:" + source_hash
    experiment["repository_commit"] = manifest["repository_commit"]
    experiment["rules_sha256"] = digest(Path(rules_path).read_bytes())
    from huggingface_hub import HfApi
    api = HfApi()
    if generator["model_revision"] is None:
        generator["model_revision"] = api.model_info(generator["model_name"]).sha
    for path, cfg in ((experiment_path, experiment), (generator_path, generator)):
        atomic_write(path, yaml.safe_dump(cfg, sort_keys=False))
    return experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="configs/experiment.yaml")
    parser.add_argument("--generator", default="configs/generator.yaml")
    parser.add_argument("--manifest", default="data/manifests/data_manifest.json")
    parser.add_argument("--rules", default="rules/rules.jsonl")
    args = parser.parse_args()
    result = initialize(args.experiment, args.generator, args.manifest, args.rules)
    print(result["run_id"])


if __name__ == "__main__":
    main()
