"""Ustala rewizje modeli i sprawdza długość reguł bez pobierania wag."""
import argparse
from pathlib import Path
import sys
import yaml
from huggingface_hub import HfApi
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.build_index import validate_rules
from src.common import atomic_write, config, digest, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolve-revisions", action="store_true")
    args = parser.parse_args()
    gen = config(ROOT / "configs/generator.yaml")
    retrieval = config(ROOT / "configs/retrieval.yaml")
    for cfg, model_key, revision_key, file in ((gen, "model_name", "model_revision", "generator.yaml"),
                                              (retrieval, "embedding_model", "embedding_model_revision", "retrieval.yaml")):
        if cfg[revision_key] is None:
            if not args.resolve_revisions:
                raise SystemExit("Fill immutable revisions or pass --resolve-revisions before protocol freeze")
            cfg[revision_key] = HfApi().model_info(cfg[model_key]).sha
            atomic_write(ROOT / "configs" / file, yaml.safe_dump(cfg, sort_keys=False))
    generator_tokenizer = AutoTokenizer.from_pretrained(gen["tokenizer_name"], revision=gen["model_revision"])
    encoder_tokenizer = AutoTokenizer.from_pretrained(retrieval["embedding_model"], revision=retrieval["embedding_model_revision"])
    lengths = validate_rules(read_jsonl(ROOT / "rules/rules.jsonl"), generator_tokenizer, encoder_tokenizer)
    write_json(ROOT / "data/manifests/rule_token_validation.json",
               {"rules_sha256": digest((ROOT / "rules/rules.jsonl").read_bytes()),
                "generator_tokenizer": gen["tokenizer_name"],
                "generator_revision": gen["model_revision"], "encoder_revision": retrieval["embedding_model_revision"],
                "special_tokens_included": False, "rule_counts": lengths})


if __name__ == "__main__":
    main()
