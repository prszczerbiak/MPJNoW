"""Dobiera reguły według największego podobieństwa okien; remisy rozstrzyga identyfikator reguły."""
import argparse
import time
import numpy as np

from .build_index import unit_vectors
from .common import ProtocolError, fingerprint, read_json, read_jsonl, write_json, write_jsonl


def windows(text, tokenizer, size=128, overlap=32):
    if size <= 0 or not 0 <= overlap < size:
        raise ProtocolError("Invalid window size or overlap")
    # Pozycje znaków zachowują oryginalny tekst, a enkoder otrzymuje wycinek tokenów.
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if not ids:
        raise ProtocolError("Cannot retrieve rules for an empty transcript")
    result = []
    for start in range(0, len(ids), size - overlap):
        end = min(start + size, len(ids))
        result.append({"window_id": len(result), "start_token": start, "end_token": end,
                       "token_count": end - start, "input_ids": ids[start:end],
                       "text": text[offsets[start][0]:offsets[end - 1][1]]})
        if end == len(ids):
            break
    return result


def encode_windows(encoder, chunks):
    import torch
    vectors = []
    encoder.eval()
    # Ponowna tokenizacja może zmienić granice WordPiece, dlatego przekazujemy identyfikatory tokenów.
    for chunk in chunks:
        ids = encoder.tokenizer.build_inputs_with_special_tokens(chunk["input_ids"])
        if len(ids) > encoder.max_seq_length:
            raise ProtocolError("Encoder would truncate a window")
        tensor = torch.tensor([ids], device=encoder.device)
        with torch.inference_mode():
            out = encoder({"input_ids": tensor, "attention_mask": torch.ones_like(tensor)})
        vectors.append(out["sentence_embedding"][0].cpu().numpy())
    return unit_vectors(vectors)


def rank_rules(encounter_id, rules, rule_vectors, chunks, window_vectors, token_counts=None, elapsed=0.0):
    if len(rules) < 4 or len({r["rule_id"] for r in rules}) != len(rules):
        raise ProtocolError("At least four distinct rules are required")
    scores = unit_vectors(window_vectors) @ unit_vectors(rule_vectors).T
    if scores.shape != (len(chunks), len(rules)):
        raise ProtocolError("Embedding and metadata shapes differ")
    best_windows = np.argmax(scores, axis=0)
    best_scores = scores[best_windows, np.arange(len(rules))]
    order = sorted(range(len(rules)), key=lambda i: (-best_scores[i], rules[i]["rule_id"]))
    result = []
    for rank, i in enumerate(order, 1):
        chunk = chunks[int(best_windows[i])]
        rid = rules[i]["rule_id"]
        result.append({"encounter_id": encounter_id, "rule_id": rid, "rank": rank,
                       "similarity_score": float(best_scores[i]), "selected_window_id": chunk["window_id"],
                       "window_text": chunk["text"], "selected": rank <= 4,
                       "retrieval_time_s": elapsed, "window_tokens": chunk["token_count"],
                       "window_start_token": chunk["start_token"], "window_end_token": chunk["end_token"],
                       **(token_counts or {}).get(rid, {})})
    return result


def retrieve(records, index):
    from sentence_transformers import SentenceTransformer
    cfg = index["config"]
    if cfg["top_k"] != 4 or cfg["similarity_metric"] != "cosine":
        raise ProtocolError("Primary protocol requires top-4 cosine retrieval")
    encoder = SentenceTransformer(cfg["embedding_model"], revision=cfg["embedding_model_revision"], device="cpu")
    result, window_index = [], []
    for record in records:
        started = time.perf_counter()
        chunks = windows(record["transcript"], encoder.tokenizer, cfg["window_size"], cfg["window_overlap"])
        vectors = encode_windows(encoder, chunks)
        result.extend(rank_rules(record["encounter_id"], index["rules"], index["vectors"], chunks, vectors,
                                 index["token_counts"], time.perf_counter() - started))
        window_index.extend({"encounter_id": record["encounter_id"], **chunk, "vector": vec.tolist()}
                            for chunk, vec in zip(chunks, vectors))
    return result, window_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/consultations.jsonl")
    parser.add_argument("--index", default="data/processed/rule_index.json")
    parser.add_argument("--split", choices=["train", "valid", "test1", "test2", "test3"], default="test1")
    parser.add_argument("--output", default="outputs/retrieval.jsonl")
    parser.add_argument("--windows-output", default="data/processed/window_index.jsonl")
    args = parser.parse_args()
    rows = [r for r in read_jsonl(args.data) if r["split"] == args.split]
    if not rows:
        raise ProtocolError("Empty retrieval split")
    index = read_json(args.index)
    result, chunks = retrieve(rows, index)
    write_jsonl(args.output, result)
    write_jsonl(args.windows_output, chunks)
    write_json(args.output + ".manifest.json", {"records_fingerprint": fingerprint(rows),
               "retrieval_fingerprint": fingerprint(result), "rules_sha256": index["rules_sha256"],
               "retrieval_config": index["config"], "index_fingerprint": fingerprint(index),
               "generator_tokenizer": index["generator_tokenizer"],
               "generator_revision": index["generator_revision"]})


if __name__ == "__main__":
    main()
