"""Lokalna ocena modelowa z kontrolą cytatów, kolejką wyjątków i audytem próbki."""
import argparse
import copy
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import yaml

import pandas as pd

from . import evaluation_prompts as prompts
from .annotate import validate_facts, validate_ratings, unique
from .common import (ProtocolError, atomic_write, config, digest, fingerprint, immutable_json,
                     read_json, read_jsonl, require_revision, write_json, write_jsonl)
from .compute_metrics import metrics
from .model_loading import GENERATOR_MODEL, loading_kwargs

METHOD = "model_assisted_v2"


class LocalJudge:
    """Osobny model ładowany po zakończeniu generowania notatek."""

    def __init__(self, cfg):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        require_revision(cfg["model_revision"])
        if cfg["model_name"] == GENERATOR_MODEL:
            raise ProtocolError("The evaluator must not be the note generator")
        self.cfg = cfg
        kwargs = loading_kwargs(cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], revision=cfg["model_revision"])
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"], revision=cfg["model_revision"], **kwargs).eval()

    def __call__(self, stage, payload):
        import torch
        from transformers import set_seed
        instruction = {"facts": prompts.FACTS, "assessment": prompts.ASSESS, "check": prompts.CHECK}[stage]
        messages = [{"role": "system", "content": prompts.SYSTEM + "\n" + instruction},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                  return_tensors="pt")
        if ids.shape[1] + self.cfg["max_new_tokens"] > self.cfg["context_budget"]:
            raise ProtocolError("Evaluator context limit exceeded; no source truncation is allowed")
        set_seed(self.cfg["seed"])
        ids = ids.to(self.model.device)
        with torch.inference_mode():
            result = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                         do_sample=False, max_new_tokens=self.cfg["max_new_tokens"],
                                         pad_token_id=self.tokenizer.eos_token_id)
        output = result[0, ids.shape[1]:]
        text = self.tokenizer.decode(output, skip_special_tokens=True)
        if len(output) >= self.cfg["max_new_tokens"] and int(output[-1]) != self.tokenizer.eos_token_id:
            raise ProtocolError("Evaluator output limit reached")
        return text


def runtime_identity():
    import torch
    return {"python": platform.python_version(), "cuda": torch.version.cuda,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            "code_sha256": fingerprint({p.name: digest(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))})}


def load_inputs(data, generations):
    rows = [r for r in data if r["split"] == "test1"]
    unique(rows, "encounter_id")
    unique(generations, "task_id")
    if not rows or not generations or any(g["split"] != "test1" for g in generations):
        raise ProtocolError("Evaluation requires nonempty test1 data and generations")
    if len({g["run_id"] for g in generations}) != 1 or len({g["environment_id"] for g in generations}) != 1:
        raise ProtocolError("Mixed generation runs or environments")
    sources = {r["encounter_id"]: r["transcript"] for r in rows}
    if any(g["encounter_id"] not in sources or g["variant"] not in ("E", "A", "B", "C")
           or g["status"] not in ("ok", "technical_failure") or not isinstance(g["response"], str)
           for g in generations):
        raise ProtocolError("Invalid generation or missing source")
    return sources


def sample_ids(rows, fraction, seed, key, strata=None):
    if not 0 < fraction <= 1:
        raise ProtocolError("Audit fraction must be positive and at most one")
    groups = {}
    for row in rows:
        groups.setdefault(row[strata] if strata else "all", []).append(row)
    selected = set()
    for group in groups.values():
        ordered = sorted(group, key=lambda r: fingerprint([seed, r[key]]))
        selected.update(r[key] for r in ordered[:max(1, math.ceil(len(group)*fraction))])
    return selected


def freeze_run(directory, data, generations, cfg, environment):
    sources = load_inputs(data, generations)
    if cfg["protocol_version"] != "model-assisted-v2" or cfg["reviewers"] != ["student_1", "student_2"]:
        raise ProtocolError("Unsupported evaluation protocol or reviewer roles")
    require_revision(cfg["model_revision"])
    generated_models = {g.get("model") for g in generations if g["variant"] != "E"}
    if cfg["model_name"] == GENERATOR_MODEL or cfg["model_name"] in generated_models:
        raise ProtocolError("The evaluator must not be the note generator")
    sample_ids([], cfg["audit_fraction"], cfg["seed"], "id")
    manifest = {"evaluation_method": METHOD, "data_hash": fingerprint(data),
                "generations_hash": fingerprint(generations), "config": cfg,
                "environment": environment, "prompt_hash": fingerprint([prompts.SYSTEM, prompts.FACTS, prompts.ASSESS, prompts.CHECK])}
    immutable_json(Path(directory) / "manifest.json", manifest)
    return sources, fingerprint(manifest)


def model_call(directory, key, stage, payload, judge):
    path = Path(directory) / "calls" / f"{key}-{stage}.json"
    if path.exists():
        saved = read_json(path)
        if saved["input_hash"] != fingerprint(payload):
            raise ProtocolError("Cached evaluator input differs; use a new evaluation directory")
    else:
        # Rezerwacja próby przed wywołaniem zapobiega powtórzeniom po przerwaniu procesu.
        saved = {"stage": stage, "input_hash": fingerprint(payload), "state": "started"}
        immutable_json(path, saved)
        try:
            saved.update(state="completed", raw=judge(stage, payload))
        except (OSError, RuntimeError, ProtocolError) as error:
            saved.update(state="failed", error=str(error))
        write_json(path, saved)
    if saved["state"] != "completed":
        raise ProtocolError(saved.get("error", "Interrupted evaluator call requires review"))
    try:
        parsed = json.loads(saved["raw"])
    except (ValueError, TypeError) as error:
        raise ProtocolError("Evaluator did not return valid JSON") from error
    if not isinstance(parsed, dict):
        raise ProtocolError("Evaluator response must be a JSON object")
    return parsed


def require_flags(value, groups):
    if not isinstance(value, dict):
        raise ProtocolError("Assessment must be an object")
    if type(value.get("review_required")) is not bool:
        raise ProtocolError("Missing review_required flag")
    for group in groups:
        if not isinstance(value.get(group), list) or not all(isinstance(x, dict) for x in value[group]):
            raise ProtocolError("Assessment groups must be lists of objects")
        for item in value[group]:
            if type(item.get("uncertain")) is not bool:
                raise ProtocolError("Missing uncertainty flag")


def validate_fact_proposal(value, source):
    require_flags(value, ["facts"])
    validate_facts(value["facts"], source)


def validate_assessment(value, facts, source, note):
    require_flags(value, ["labels", "coverage"])
    if not isinstance(value.get("claims"), list) or not all(isinstance(x, dict) for x in value["claims"]):
        raise ProtocolError("Claims must be a list of objects")
    unique(value["claims"], "claim_id")
    for claim in value["claims"]:
        if not claim["text"].strip() or not claim["note_span"].strip() or claim["note_span"] not in note:
            raise ProtocolError("Every claim must quote the original note")
    validate_ratings(value, value["claims"], facts, source)
    metrics([x["label"] for x in value["labels"]], [x["covered"] for x in value["coverage"]], value["response_kind"])
    if (not note.strip()) != (value["response_kind"] == "empty"):
        raise ProtocolError("Empty-response classification does not match the text")


def uncertain(value, groups):
    return value["review_required"] or any(x["uncertain"] for group in groups for x in value[group])


def decision_signature(value):
    return {"labels": sorted((r["claim_id"], r["label"]) for r in value["labels"]),
            "coverage": sorted((r["fact_id"], r["covered"], sorted(r["claim_ids"])) for r in value["coverage"])}


def queue_review(directory, key, kind, source, proposal, reasons, **extra):
    # Pakiet nie zawiera wariantu, ziarna ani identyfikatora zadania generatora.
    packet = {"review_id": key, "kind": kind, "transcript": source,
              "proposal": proposal, "reasons": sorted(set(reasons)), **extra}
    immutable_json(Path(directory) / "queue" / f"{key}.json", packet)


def fact_key(eid):
    return "F" + fingerprint(eid)[:20]


def note_key(task_id):
    return "N" + fingerprint(task_id)[:20]


def evaluate_facts(data, generations, cfg, judge, directory, environment):
    directory = Path(directory)
    sources, evaluation_id = freeze_run(directory, data, generations, cfg, environment)
    encounters = sorted({g["encounter_id"] for g in generations if g["status"] == "ok"})
    audit = sample_ids([{"id": eid} for eid in encounters], cfg["audit_fraction"], cfg["seed"], "id")
    for eid in encounters:
        key = fact_key(eid)
        if (directory / "facts" / f"{key}.json").exists() or (directory / "queue" / f"{key}.json").exists():
            continue
        proposal, reasons = None, []
        try:
            proposal = model_call(directory, key, "facts", {"transcript": sources[eid]}, judge)
            validate_fact_proposal(proposal, sources[eid])
            if uncertain(proposal, ["facts"]):
                reasons.append("uncertain_facts")
        except (ProtocolError, KeyError, TypeError, ValueError, AttributeError) as error:
            reasons.append("invalid_or_failed_proposal: " + str(error))
        if eid in audit:
            reasons.append("random_audit")
        if reasons:
            queue_review(directory, key, "facts", sources[eid], proposal, reasons)
        else:
            immutable_json(directory / "facts" / f"{key}.json",
                           {"value": proposal, "origin": "automatic", "evaluation_id": evaluation_id})


def evaluate_notes(data, generations, cfg, judge, directory, environment):
    directory = Path(directory)
    sources, evaluation_id = freeze_run(directory, data, generations, cfg, environment)
    successful = [g for g in generations if g["status"] == "ok"]
    if any(not (directory / "facts" / f'{fact_key(g["encounter_id"])}.json').exists() for g in successful):
        raise ProtocolError("Resolve fact reviews before evaluating any notes")
    audit = sample_ids(successful, cfg["audit_fraction"], cfg["seed"], "task_id", "variant")
    immutable_json(directory / "private" / "mapping.json",
                   [{"review_id": note_key(g["task_id"]), "task_id": g["task_id"]} for g in generations])
    for g in successful:
        key, eid = note_key(g["task_id"]), g["encounter_id"]
        if (directory / "decisions" / f"{key}.json").exists() or (directory / "queue" / f"{key}.json").exists():
            continue
        fact_record = read_json(directory / "facts" / f"{fact_key(eid)}.json")
        facts = fact_record["value"]["facts"]
        proposal, check, reasons = None, None, []
        payload = {"transcript": sources[eid], "note": g["response"], "facts": facts}
        try:
            if not g["response"].strip():
                proposal = {"response_kind": "empty", "claims": [], "labels": [], "review_required": False,
                            "coverage": [{"fact_id": f["fact_id"], "covered": False, "claim_ids": [],
                                          "rationale": "Empty response", "uncertain": False} for f in facts]}
                check = copy.deepcopy(proposal)
            else:
                proposal = model_call(directory, key, "assessment", payload, judge)
                validate_assessment(proposal, facts, sources[eid], g["response"])
                # Drugi przebieg widzi twierdzenia, lecz nie etykiety ani uzasadnienia pierwszego.
                check = model_call(directory, key, "check",
                                   {**payload, "facts": list(reversed(facts)), "claims": proposal["claims"]}, judge)
                check = {**check, "claims": proposal["claims"], "response_kind": proposal["response_kind"]}
            validate_assessment(proposal, facts, sources[eid], g["response"])
            validate_assessment(check, facts, sources[eid], g["response"])
            if decision_signature(proposal) != decision_signature(check):
                reasons.append("inconsistent_checks")
            if uncertain(proposal, ["labels", "coverage"]) or uncertain(check, ["labels", "coverage"]):
                reasons.append("uncertain_assessment")
            if g["response"].strip() and proposal["response_kind"] != "clinical":
                reasons.append("nonclinical_response")
        except (ProtocolError, KeyError, TypeError, ValueError, AttributeError) as error:
            reasons.append("invalid_or_failed_assessment: " + str(error))
        if g["task_id"] in audit:
            reasons.append("random_audit")
        if reasons:
            queue_review(directory, key, "note", sources[eid], proposal, reasons,
                         response=g["response"], facts=facts, fact_hash=fingerprint(fact_record), second_check=check)
        else:
            immutable_json(directory / "decisions" / f"{key}.json",
                           {"value": proposal, "origin": "automatic", "evaluation_id": evaluation_id,
                            "fact_hash": fingerprint(fact_record)})


def submit_review(directory, review_id, reviewer, value, rationale):
    directory = Path(directory)
    if not re.fullmatch(r"[FN][0-9a-f]{20}", review_id):
        raise ProtocolError("Invalid review identifier")
    manifest = read_json(directory / "manifest.json")
    if reviewer not in manifest["config"]["reviewers"] or not rationale.strip():
        raise ProtocolError("A named reviewer and rationale are required")
    packet = read_json(directory / "queue" / f"{review_id}.json")
    if packet["kind"] == "facts":
        validate_fact_proposal(value, packet["transcript"])
        if uncertain(value, ["facts"]):
            raise ProtocolError("Resolve uncertainty before accepting facts")
        destination = "facts"
    else:
        validate_assessment(value, packet["facts"], packet["transcript"], packet["response"])
        if uncertain(value, ["labels", "coverage"]):
            raise ProtocolError("Resolve uncertainty before accepting an assessment")
        destination = "decisions"
    record = {"value": value, "origin": "human_review", "reviewer": reviewer, "rationale": rationale,
              "packet_hash": fingerprint(packet), "evaluation_id": fingerprint(manifest)}
    if packet["kind"] == "note":
        record["fact_hash"] = packet["fact_hash"]
    immutable_json(directory / destination / f"{review_id}.json", record)


def pending_reviews(directory):
    directory = Path(directory)
    return [p for p in sorted((directory / "queue").glob("*.json"))
            if not (directory / ("facts" if p.stem.startswith("F") else "decisions") / p.name).exists()]


def review_interactively(directory, reviewer):
    for path in pending_reviews(directory):
        packet = read_json(path)
        print(json.dumps(packet, ensure_ascii=False, indent=2))
        choice = input("[a] zatwierdź, [e] wczytaj poprawiony JSON, [q] zakończ: ").strip().lower()
        if choice == "q":
            return
        if choice == "a":
            value = copy.deepcopy(packet["proposal"])
            if not isinstance(value, dict):
                print("Brak poprawnej propozycji. Wymagany jest poprawiony plik JSON.")
                continue
            value["review_required"] = False
            for group in ("facts", "labels", "coverage"):
                for item in value.get(group, []):
                    item["uncertain"] = False
        elif choice == "e":
            value = read_json(input("Ścieżka poprawionego JSON: ").strip())
        else:
            continue
        rationale = input("Uzasadnienie decyzji (po angielsku): ").strip()
        try:
            submit_review(directory, path.stem, reviewer, value, rationale)
        except (ProtocolError, KeyError, TypeError, ValueError, AttributeError) as error:
            print(f"Nie zapisano decyzji: {error}")


def export_metrics(data, generations, directory):
    directory = Path(directory)
    sources = load_inputs(data, generations)
    manifest = read_json(directory / "manifest.json")
    if manifest["data_hash"] != fingerprint(data) or manifest["generations_hash"] != fingerprint(generations):
        raise ProtocolError("Evaluation inputs changed")
    if pending_reviews(directory):
        raise ProtocolError("Pending reviews block final metrics; no cases are silently omitted")
    result, origins = [], {"automatic": 0, "human_review": 0}
    for g in generations:
        base = {k: g[k] for k in ("run_id", "task_id", "encounter_id", "variant", "seed", "split", "environment_id", "status")}
        base.update(evaluation_method=METHOD, evaluation_id=fingerprint(manifest))
        if g["status"] != "ok":
            result.append(base)
            continue
        decision = read_json(directory / "decisions" / f'{note_key(g["task_id"])}.json')
        fact_record = read_json(directory / "facts" / f'{fact_key(g["encounter_id"])}.json')
        if (decision["fact_hash"] != fingerprint(fact_record) or decision["evaluation_id"] != fingerprint(manifest)
                or fact_record["evaluation_id"] != fingerprint(manifest)):
            raise ProtocolError("Evaluation provenance mismatch")
        value = decision["value"]
        validate_assessment(value, fact_record["value"]["facts"], sources[g["encounter_id"]], g["response"])
        origins[decision["origin"]] += 1
        result.append({**base, **metrics([x["label"] for x in value["labels"]],
                                         [x["covered"] for x in value["coverage"]], value["response_kind"]),
                       "assessment_origin": decision["origin"], "assessment_hash": fingerprint(decision)})
    frame = pd.DataFrame(result)
    atomic_write(directory / "metrics.csv", frame.to_csv(index=False))
    audits = {"facts": {"reviewed": 0, "corrected": 0}, "note": {"reviewed": 0, "corrected": 0}}
    for path in (directory / "queue").glob("*.json"):
        packet = read_json(path)
        if "random_audit" not in packet["reasons"]:
            continue
        kind = packet["kind"]
        final = read_json(directory / ("facts" if kind == "facts" else "decisions") / path.name)["value"]
        original = packet["proposal"]
        if kind == "facts":
            def signature(value):
                return [(f["fact_id"], f["text"], f["source_span"], f["category"]) for f in value["facts"]]
        else:
            def signature(value):
                return {**decision_signature(value), "claims": value["claims"], "response_kind": value["response_kind"]}
        try:
            changed = signature(final) != signature(original)
        except (KeyError, TypeError, ValueError):
            changed = True
        audits[kind]["reviewed"] += 1
        audits[kind]["corrected"] += int(changed)
    write_json(directory / "summary.json", {"evaluation_method": METHOD, "inference_scope": "exploratory",
                                            "notes_by_origin": origins, "pending_reviews": 0,
                                            "random_audit": audits,
                                            "human_interrater_agreement": None})
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["pin", "facts", "notes", "review", "export", "status"])
    parser.add_argument("--data", default="data/processed/consultations.jsonl")
    parser.add_argument("--generations", default="outputs/generations.jsonl")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--directory")
    parser.add_argument("--reviewer", choices=["student_1", "student_2"])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cfg = config(args.config)
    directory = args.directory or cfg["output_directory"]
    if args.action == "pin":
        from huggingface_hub import HfApi
        if (Path(directory) / "manifest.json").exists():
            raise ProtocolError("Cannot change evaluator revision after evaluation freeze")
        if cfg["model_revision"] is None:
            cfg["model_revision"] = HfApi().model_info(cfg["model_name"]).sha
            require_revision(cfg["model_revision"])
            atomic_write(args.config, yaml.safe_dump(cfg, sort_keys=False))
        print(cfg["model_revision"])
    elif args.action == "status":
        print(json.dumps({"pending_reviews": [p.stem for p in pending_reviews(directory)]}))
    elif args.action == "review":
        if not args.reviewer:
            parser.error("--reviewer is required")
        review_interactively(directory, args.reviewer)
    else:
        data, generations = read_jsonl(args.data), read_jsonl(args.generations)
        if args.action == "export":
            export_metrics(data, generations, directory)
        else:
            if not args.execute:
                parser.error("Model evaluation requires --execute")
            environment = runtime_identity()
            freeze_run(directory, data, generations, cfg, environment)
            judge = LocalJudge(cfg)
            {"facts": evaluate_facts, "notes": evaluate_notes}[args.action](
                data, generations, cfg, judge, directory, environment)


if __name__ == "__main__":
    main()
