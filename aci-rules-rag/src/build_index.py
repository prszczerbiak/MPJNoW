"""Sprawdza reguły i buduje indeks wektorowy dla ustalonej rewizji modelu."""
import argparse
from pathlib import Path
import numpy as np

from .common import ProtocolError, config, digest, read_jsonl, require_revision, write_json

FIXED = {
    "G01": "Preserve negations and do not reverse the presence or absence of a symptom.",
    "G02": "Do not equate an absence of mention with a normal result or absence of a condition.",
    "G03": "Do not add a medication name, dose, unit, route, or administration frequency.",
    "G04": "Preserve the level of certainty and do not present a suspicion as a diagnosis.",
}
TAGS = {"negation", "uncertainty", "medications", "doses_units", "chronology",
        "information_subject", "diagnosis_causality", "plan_actions"}


def validate_rules(rules, generator_tokenizer=None, encoder_tokenizer=None):
    expected = {f"G{i:02}" for i in range(1, 13)} | {f"S{i:02}" for i in range(1, 49)}
    if len(rules) != 60 or {r["rule_id"] for r in rules} != expected:
        raise ProtocolError("Exactly G01-G12 and S01-S48 are required")
    lengths = {}
    for rule in rules:
        required = {"rule_id", "text", "tags", "rationale", "source_inspiration", "version"}
        if not required <= rule.keys() or any(not rule[k] for k in required):
            raise ProtocolError("Incomplete rule")
        if not isinstance(rule["tags"], list) or not set(rule["tags"]) <= TAGS:
            raise ProtocolError("Invalid rule tags")
        rid = rule["rule_id"]
        if rid in FIXED and rule["text"] != FIXED[rid]:
            raise ProtocolError(f"Fixed rule changed: {rid}")
        lengths[rid] = {}
        for name, tokenizer, limit in (("generator_tokens", generator_tokenizer, 96),
                                       ("encoder_tokens", encoder_tokenizer, 128)):
            if tokenizer is not None:
                count = len(tokenizer.encode(rule["text"], add_special_tokens=False))
                lengths[rid][name] = count
                if count > limit:
                    raise ProtocolError(f"{rid} exceeds {name} limit: {count}")
    if set().union(*(set(r["tags"]) for r in rules)) != TAGS:
        raise ProtocolError("Rules must cover all eight domains")
    return lengths


def unit_vectors(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ProtocolError("Invalid embedding matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ProtocolError("Zero embedding")
    return matrix / norms


def build(rules_path, retrieval_config, generator_config, output):
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    require_revision(retrieval_config["embedding_model_revision"])
    require_revision(generator_config["model_revision"])
    rules = sorted(read_jsonl(rules_path), key=lambda r: r["rule_id"])
    encoder = SentenceTransformer(retrieval_config["embedding_model"],
                                  revision=retrieval_config["embedding_model_revision"], device="cpu")
    tokenizer = AutoTokenizer.from_pretrained(generator_config["tokenizer_name"],
                                              revision=generator_config["model_revision"])
    lengths = validate_rules(rules, tokenizer, encoder.tokenizer)
    vectors = unit_vectors(encoder.encode([r["text"] for r in rules], convert_to_numpy=True,
                                          normalize_embeddings=True))
    if vectors.shape[1] != retrieval_config["embedding_dimension"]:
        raise ProtocolError("Embedding dimension does not match configuration")
    index = {"rules": rules, "vectors": vectors.tolist(), "token_counts": lengths,
             "rules_sha256": digest(Path(rules_path).read_bytes()), "config": retrieval_config,
             "generator_tokenizer": generator_config["tokenizer_name"],
             "generator_revision": generator_config["model_revision"]}
    write_json(output, index)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", default="rules/rules.jsonl")
    parser.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    parser.add_argument("--generator-config", default="configs/generator.yaml")
    parser.add_argument("--output", default="data/processed/rule_index.json")
    args = parser.parse_args()
    build(args.rules, config(args.retrieval_config), config(args.generator_config), args.output)


if __name__ == "__main__":
    main()
