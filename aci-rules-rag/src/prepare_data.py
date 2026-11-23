"""Wczytuje i sprawdza oficjalne pliki CSV bez zmiany treści klinicznej."""
import argparse
import csv
from pathlib import Path
import subprocess
import unicodedata

from .common import ProtocolError, digest, write_json, write_jsonl

OFFICIAL = "https://github.com/wyim/aci-bench.git"
COUNTS = {"train": 67, "valid": 20, "test1": 40, "test2": 40, "test3": 40}
# Nazwy pochodzą z oficjalnego repozytorium; katalogi są wyszukiwane.
BASENAMES = {"train": "train.csv", "valid": "valid.csv",
             "test1": "clinicalnlp_taskB_test1.csv",
             "test2": "clinicalnlp_taskC_test2.csv", "test3": "clef_taskC_test3.csv"}


def normalize(text):
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def load_csv(path, split):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"encounter_id", "dataset", "dialogue", "note"} <= set(reader.fieldnames or []):
            raise ProtocolError(f"Unexpected CSV schema: {path}")
        rows = []
        for row in reader:
            if None in row or any(not isinstance(row.get(k), str) or not row[k].strip()
                                  for k in ("encounter_id", "dataset", "dialogue", "note")):
                raise ProtocolError(f"Empty or malformed record in {path}")
            if any("\ufffd" in row[k] or "\x00" in row[k] for k in row):
                raise ProtocolError(f"Corrupted text in {path}")
            if row["encounter_id"] != row["encounter_id"].strip():
                raise ProtocolError("Whitespace in encounter identifier")
            rows.append({"encounter_id": row["encounter_id"], "split": split,
                         "source_type": row["dataset"], "transcript": normalize(row["dialogue"]),
                         "reference_note": normalize(row["note"])})
        return rows


def validate_records(rows, expected=COUNTS):
    ids, transcripts, references = {}, {}, {}
    counts = {s: 0 for s in expected}
    duplicate_texts = []
    for row in rows:
        eid, split = row["encounter_id"], row["split"]
        if split not in counts or eid in ids:
            raise ProtocolError(f"Duplicate encounter or unexpected split: {eid}, {split}")
        ids[eid] = split
        counts[split] += 1
        for field, seen in (("transcript", transcripts), ("reference_note", references)):
            if not row[field].strip():
                raise ProtocolError(f"Empty {field}: {eid}")
            key = digest(row[field])
            if key in seen:
                duplicate_texts.append({"field": field, "encounters": [seen[key], eid]})
            else:
                seen[key] = eid
    if counts != expected:
        raise ProtocolError(f"Split counts differ: {counts}, expected {expected}")
    return {"counts": counts, "duplicate_texts": duplicate_texts,
            "normalization": "Unicode NFC and LF only", "unique_encounter_ids": len(ids)}


def prepare(repository, output, manifests):
    repository, manifests = Path(repository), Path(manifests)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repository), *args], text=True).strip()
    if git("remote", "get-url", "origin").removesuffix(".git") != OFFICIAL.removesuffix(".git"):
        raise ProtocolError("Only the official ACI-Bench repository is allowed")
    # Porównanie dopuszcza różnice CRLF/LF wynikające z pobrania plików w Windows.
    # Wyjątek dotyczy tłumaczeń README i SETUP; pozostałe pliki podlegają kontroli.
    clean = subprocess.run(["git", "-C", str(repository), "diff", "--quiet",
                            "--ignore-cr-at-eol", "HEAD", "--", ".", ":(exclude)README.md", ":(exclude)SETUP.md"], check=False)
    if clean.returncode:
        raise ProtocolError("Official repository has modified tracked files")
    commit = git("rev-parse", "HEAD")
    rows, files = [], []
    for split, basename in BASENAMES.items():
        candidates = []
        for path in repository.rglob(basename):
            with path.open(encoding="utf-8-sig", newline="") as stream:
                fields = next(csv.reader(stream), [])
            if {"encounter_id", "dataset", "dialogue", "note"} <= set(fields):
                candidates.append(path)
        if len(candidates) != 1:
            raise ProtocolError(f"Expected one full CSV for {split}; found {candidates}")
        path = candidates[0]
        rows.extend(load_csv(path, split))
        files.append({"split": split, "path": path.relative_to(repository).as_posix(),
                      "sha256": digest(path.read_bytes())})
    report = validate_records(rows)
    manifest = {"repository": OFFICIAL, "repository_commit": commit, "files": files, **report}
    write_json(manifests / "data_manifest.json", manifest)
    if report["duplicate_texts"]:
        raise ProtocolError("Text duplicates detected; inspect manifest and amend protocol, never silently drop")
    # Notatki referencyjne są zapisywane osobno, poza wejściem generatora.
    write_jsonl(Path(output) / "consultations.jsonl",
                [{k: v for k, v in r.items() if k != "reference_note"} for r in rows])
    write_jsonl(Path(output) / "references.jsonl",
                [{k: r[k] for k in ("encounter_id", "split", "reference_note")} for r in rows])
    manifest["consultations_sha256"] = digest((Path(output) / "consultations.jsonl").read_bytes())
    manifest["references_sha256"] = digest((Path(output) / "references.jsonl").read_bytes())
    write_json(manifests / "data_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="data/raw/aci-bench")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--manifests", default="data/manifests")
    args = parser.parse_args()
    prepare(args.repository, args.output, args.manifests)


if __name__ == "__main__":
    main()
