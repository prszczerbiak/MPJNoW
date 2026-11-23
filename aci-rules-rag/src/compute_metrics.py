"""Oblicza metryki na podstawie uzgodnionych etykiet i ocen odtworzenia faktów."""
import argparse
from pathlib import Path
import pandas as pd

from .common import ProtocolError, atomic_write, fingerprint, read_json, read_jsonl


def metrics(labels, coverage, response_kind="clinical"):
    if any(label not in ("P", "S", "N") for label in labels):
        raise ProtocolError("Labels must be P, S, or N")
    if not coverage or any(type(value) is not bool for value in coverage):
        raise ProtocolError("K must be positive and coverage must contain human booleans")
    if response_kind not in ("clinical", "empty", "refusal", "no_clinical_claims"):
        raise ProtocolError("Unknown response type")
    if bool(labels) != (response_kind == "clinical") or (not labels and any(coverage)):
        raise ProtocolError("Response type, claims, and coverage disagree")
    p, s, n = (labels.count(label) for label in ("P", "S", "N"))
    total, k = len(labels), len(coverage)
    h = (s+n)/total if total else None
    c = sum(coverage)/k
    return {"P": p, "S": s, "N": n, "T": total, "K": k, "H": h,
            "H_star": h if total else 1.0, "C": c, "O": 1-c}


def compute(generations, directory):
    directory = Path(directory)
    mapping = {r["task_id"]: r for r in read_json(directory / "private" / "blinding_key.json")}
    rows = []
    exposure = read_json(directory / "private" / "exposure_manifest.json")
    if fingerprint(generations) != exposure["generation_hash"]:
        raise ProtocolError("Generations differ from the blinded annotation release")
    for row in generations:
        if row["split"] != "test1":
            continue
        base = {k: row[k] for k in ("run_id", "task_id", "encounter_id", "variant", "seed", "split", "environment_id", "status")}
        base["evaluation_method"] = "human"
        if row["status"] != "ok":
            rows.append(base)
            continue
        bid = mapping[row["task_id"]]["blind_id"]
        decision = read_json(directory / "adjudicated" / f"{bid}.json")
        if decision["facts_hash"] != exposure["fact_hashes"][row["encounter_id"]]:
            raise ProtocolError("Fact list changed after note exposure")
        result = metrics([x["label"] for x in decision["labels"]],
                         [x["covered"] for x in decision["coverage"]], decision["response_kind"])
        rows.append({**base, **result, "adjudication_hash": fingerprint(decision)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", default="outputs/generations.jsonl")
    parser.add_argument("--annotations", default="annotations")
    parser.add_argument("--output", default="outputs/metrics.csv")
    args = parser.parse_args()
    atomic_write(args.output, compute(read_jsonl(args.generations), args.annotations).to_csv(index=False))


if __name__ == "__main__":
    main()
