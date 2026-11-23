"""Przygotowuje listy faktów, a po ich zatwierdzeniu udostępnia notatki bez oznaczeń wariantu."""
import argparse
import random
from pathlib import Path

from .common import ProtocolError, fingerprint, immutable_json, read_json, read_jsonl, write_jsonl

CATEGORIES = ["visit_reason", "symptoms", "negations", "medications", "results_tests", "diagnoses_uncertainty", "plan"]


def prepare_facts(records, directory):
    records = [r for r in records if r["split"] == "test1"]
    if not records:
        raise ProtocolError("Fact preparation requires test1 consultations")
    directory = Path(directory)
    for r in records:
        source = {k: r[k] for k in ("encounter_id", "split", "transcript")}
        immutable_json(directory / "sources" / f'{r["encounter_id"]}.json', source)
        for annotator in ("annotator_1", "annotator_2"):
            immutable_json(directory / "packets" / annotator / "facts" / f'{r["encounter_id"]}.json',
                           {**source, "annotator_id": annotator, "categories": CATEGORIES, "facts": []})


def prepare_notes(generations, directory, seed=27182):
    directory = Path(directory)
    rows = [r for r in generations if r["split"] == "test1" and r["status"] == "ok"]
    if not rows:
        raise ProtocolError("No successful test1 generations")
    # Katalogi private/ i submissions/ pozostają u koordynatora, poza pakietami anotatorów.
    facts_hashes = {}
    for eid in sorted({r["encounter_id"] for r in rows}):
        path = directory / "frozen_facts" / f"{eid}.json"
        if not path.exists():
            raise ProtocolError(f"Freeze required facts before exposing any note: {eid}")
        facts_hashes[eid] = fingerprint(read_json(path))
    shuffled = sorted(rows, key=lambda r: r["task_id"])
    random.Random(seed).shuffle(shuffled)
    mapping = []
    for position, row in enumerate(shuffled):
        blind_id = "N" + fingerprint([seed, row["task_id"]])[:20]
        eid = row["encounter_id"]
        packet = {"blind_id": blind_id, "encounter_id": eid,
                  "transcript": read_json(directory / "sources" / f"{eid}.json")["transcript"],
                  "response": row["response"], "facts": read_json(directory / "frozen_facts" / f"{eid}.json")["facts"],
                  "claims": [], "response_kind": None,
                  "instructions": "Manually segment all clinical assertions; do not label sentence proposals automatically."}
        immutable_json(directory / "segmentation" / f"{blind_id}.json", packet)
        mapping.append({"blind_id": blind_id, "task_id": row["task_id"], "encounter_id": eid,
                        "variant": row["variant"], "seed": row["seed"], "position": position})
    immutable_json(directory / "private" / "blinding_key.json", mapping)
    immutable_json(directory / "private" / "exposure_manifest.json",
                   {"fact_hashes": facts_hashes, "generation_hash": fingerprint(generations), "seed": seed})
    write_jsonl(directory / "private" / "annotation_order.jsonl", mapping)
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["facts", "notes"])
    parser.add_argument("--data", default="data/processed/consultations.jsonl")
    parser.add_argument("--generations", default="outputs/generations.jsonl")
    parser.add_argument("--directory", default="annotations")
    args = parser.parse_args()
    if args.stage == "facts":
        prepare_facts(read_jsonl(args.data), args.directory)
    else:
        prepare_notes(read_jsonl(args.generations), args.directory)


if __name__ == "__main__":
    main()
