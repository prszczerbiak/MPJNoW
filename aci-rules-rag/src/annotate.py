"""Sprawdza i zapisuje niezależne oceny anotatorów oraz rozstrzygnięcia sporów."""
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from .common import ProtocolError, fingerprint, immutable_json, read_json, write_json
from .prepare_annotation import CATEGORIES

LABELS = ["P", "S", "N"]
ANNOTATORS = {"annotator_1", "annotator_2"}


def unique(items, field):
    ids = [r[field] for r in items]
    if len(ids) != len(set(ids)) or any(not isinstance(i, str) or not i.strip() for i in ids):
        raise ProtocolError(f"Non-unique or empty {field}")
    return set(ids)


def validate_facts(facts, source):
    if not facts:
        raise ProtocolError("K must be positive; empty fact sets require protocol review")
    unique(facts, "fact_id")
    for fact in facts:
        if fact["category"] not in CATEGORIES or not fact["text"].strip() or not fact["rationale"].strip():
            raise ProtocolError("Fact needs category, text, and relevance rationale")
        if not fact["source_span"].strip() or fact["source_span"] not in source:
            raise ProtocolError("Every fact must cite an exact transcript span")


def submit_facts(directory, submission):
    directory = Path(directory)
    eid, aid = submission["encounter_id"], submission["annotator_id"]
    if aid not in ANNOTATORS:
        raise ProtocolError("Two independent primary annotators are required")
    if (directory / "private" / "exposure_manifest.json").exists():
        raise ProtocolError("Fact submissions cannot change after notes have been exposed")
    source = read_json(directory / "sources" / f"{eid}.json")
    validate_facts(submission["facts"], source["transcript"])
    immutable_json(directory / "submissions" / aid / "facts" / f"{eid}.json", submission)


def freeze_facts(directory, decision):
    directory = Path(directory)
    eid = decision["encounter_id"]
    if decision["adjudicator_id"] != "annotator_3" or not decision["rationale"].strip():
        raise ProtocolError("A third annotator must sign and justify the fact consensus")
    if (directory / "private" / "exposure_manifest.json").exists():
        raise ProtocolError("Required facts must be frozen before note exposure")
    independent = [read_json(directory / "submissions" / aid / "facts" / f"{eid}.json")
                   for aid in sorted(ANNOTATORS)]
    source = read_json(directory / "sources" / f"{eid}.json")
    validate_facts(decision["facts"], source["transcript"])
    immutable_json(directory / "frozen_facts" / f"{eid}.json",
                   {**decision, "independent_hashes": [fingerprint(s) for s in independent],
                    "source_hash": fingerprint(source)})


def freeze_claims(directory, decision):
    directory = Path(directory)
    bid = decision["blind_id"]
    packet = read_json(directory / "segmentation" / f"{bid}.json")
    if decision["reviewed_by"] != ["annotator_1", "annotator_2"]:
        raise ProtocolError("Both annotators must review atomic segmentation before independent labels")
    claims = decision["claims"]
    unique(claims, "claim_id")
    kind = decision["response_kind"]
    if kind not in ("clinical", "empty", "refusal", "no_clinical_claims"):
        raise ProtocolError("Invalid response kind")
    if bool(claims) != (kind == "clinical"):
        raise ProtocolError("Only clinical responses have claims")
    if kind == "empty" and packet["response"].strip():
        raise ProtocolError("Nonempty response cannot be classified as empty")
    if not packet["response"].strip() and kind != "empty":
        raise ProtocolError("Blank responses must be classified as empty")
    for claim in claims:
        if not claim["text"].strip() or not claim["note_span"].strip() or claim["note_span"] not in packet["response"]:
            raise ProtocolError("Atomic claims need text and an exact note span")
    frozen = {**decision, "encounter_id": packet["encounter_id"], "packet_hash": fingerprint(packet)}
    immutable_json(directory / "frozen_claims" / f"{bid}.json", frozen)
    for aid in sorted(ANNOTATORS):
        immutable_json(directory / "packets" / aid / "notes" / f"{bid}.json",
                       {**packet, "claims": claims, "response_kind": kind, "annotator_id": aid,
                        "labels": [], "coverage": []})


def validate_ratings(rating, claims, facts, source):
    claim_ids = {c["claim_id"] for c in claims}
    fact_ids = {f["fact_id"] for f in facts}
    if unique(rating["labels"], "claim_id") != claim_ids or unique(rating["coverage"], "fact_id") != fact_ids:
        raise ProtocolError("Rate every claim and every required fact exactly once")
    labels = {}
    for item in rating["labels"]:
        if item["label"] not in LABELS or not item["rationale"].strip():
            raise ProtocolError("Each label needs P/S/N and a rationale")
        evidence = item.get("source_span", "")
        if item["label"] in ("P", "S") and (not evidence.strip() or evidence not in source):
            raise ProtocolError("P and S require exact supporting or contradicting source evidence")
        labels[item["claim_id"]] = item["label"]
    for item in rating["coverage"]:
        if type(item["covered"]) is not bool or not item["rationale"].strip():
            raise ProtocolError("Coverage must be an explicit human boolean with rationale")
        linked = item["claim_ids"]
        if len(linked) != len(set(linked)) or not set(linked) <= claim_ids:
            raise ProtocolError("Coverage links must identify distinct existing claims")
        if item["covered"] and (not linked or any(labels[c] != "P" for c in linked)):
            raise ProtocolError("Covered facts require supported claims")


def materials(directory, bid):
    frozen = read_json(directory / "frozen_claims" / f"{bid}.json")
    eid = frozen["encounter_id"]
    facts = read_json(directory / "frozen_facts" / f"{eid}.json")["facts"]
    source = read_json(directory / "sources" / f"{eid}.json")["transcript"]
    return frozen, facts, source


def submit_rating(directory, rating):
    directory = Path(directory)
    bid, aid = rating["blind_id"], rating["annotator_id"]
    if aid not in ANNOTATORS:
        raise ProtocolError("Invalid independent annotator")
    frozen, facts, source = materials(directory, bid)
    validate_ratings(rating, frozen["claims"], facts, source)
    immutable_json(directory / "submissions" / aid / "notes" / f"{bid}.json", rating)


def agreement(first, second):
    left = {r["claim_id"]: r["label"] for r in first}
    right = {r["claim_id"]: r["label"] for r in second}
    if len(left) != len(first) or len(right) != len(second) or left.keys() != right.keys():
        raise ProtocolError("Agreement requires exactly aligned claim IDs")
    ids = sorted(left)
    a, b = [left[k] for k in ids], [right[k] for k in ids]
    if not set(a+b) <= set(LABELS):
        raise ProtocolError("Unknown label")
    if not ids:
        return {"n": 0, "raw_agreement": None, "cohen_kappa": None, "labels": LABELS,
                "disagreement_matrix": [[0]*3 for _ in range(3)]}
    kappa = cohen_kappa_score(a, b, labels=LABELS) if len(set(a+b)) > 1 else np.nan
    return {"n": len(ids), "raw_agreement": float(np.mean(np.array(a) == np.array(b))),
            "cohen_kappa": float(kappa) if np.isfinite(kappa) else None, "labels": LABELS,
            "disagreement_matrix": confusion_matrix(a, b, labels=LABELS).tolist()}


def adjudicate(directory, decision):
    directory = Path(directory)
    bid = decision["blind_id"]
    if decision["adjudicator_id"] != "annotator_3" or not decision["rationale"].strip():
        raise ProtocolError("Third-annotator decision and rationale required")
    independent = [read_json(directory / "submissions" / aid / "notes" / f"{bid}.json")
                   for aid in sorted(ANNOTATORS)]
    frozen, facts, source = materials(directory, bid)
    validate_ratings(decision, frozen["claims"], facts, source)
    pre = agreement(independent[0]["labels"], independent[1]["labels"])
    immutable_json(directory / "agreement" / f"{bid}.json", pre)
    immutable_json(directory / "adjudicated" / f"{bid}.json",
                   {**decision, "response_kind": frozen["response_kind"], "encounter_id": frozen["encounter_id"],
                    "K": len(facts), "independent_hashes": [fingerprint(r) for r in independent],
                    "facts_hash": fingerprint(read_json(directory / "frozen_facts" / f'{frozen["encounter_id"]}.json'))})


def export_agreement(directory, output):
    directory = Path(directory)
    first, second = [], []
    for entry in read_json(directory / "private" / "blinding_key.json"):
        bid = entry["blind_id"]
        for aid, target in (("annotator_1", first), ("annotator_2", second)):
            for label in read_json(directory / "submissions" / aid / "notes" / f"{bid}.json")["labels"]:
                target.append({**label, "claim_id": bid + ":" + label["claim_id"]})
    write_json(output, agreement(first, second))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["submit-facts", "freeze-facts", "freeze-claims", "submit-rating", "adjudicate", "agreement"])
    parser.add_argument("file", help="Input JSON; output JSON for agreement")
    parser.add_argument("--directory", default="annotations")
    args = parser.parse_args()
    if args.action == "agreement":
        export_agreement(args.directory, args.file)
    else:
        {"submit-facts": submit_facts, "freeze-facts": freeze_facts, "freeze-claims": freeze_claims,
         "submit-rating": submit_rating, "adjudicate": adjudicate}[args.action](args.directory, read_json(args.file))


if __name__ == "__main__":
    main()
