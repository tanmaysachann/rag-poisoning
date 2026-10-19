"""Question-only full-corpus retrieval, then separately score frozen predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from data.build_full_corpus_v17 import ROOT, normalized
from data.build_trusted_snapshot_v12 import BEIR_MD5, digest
from evaluation.evaluate_trusted_snapshot_v12 import policy_prediction
from generation.trusted_reference import ReferenceReader
from retrieval.trusted_snapshot_v12 import TrustedSnapshotSearch


SPLIT = ROOT / "research_protocols" / "full_corpus_v17_matched_split.json"
LOCAL = ROOT / "data" / "full_corpus_v17"
REPORT = ROOT / "results" / "full_corpus_v17.json"
QA_PATH = ROOT / "artifacts" / "vendor" / "roberta-base-squad2-nq"
QA_SHA256 = "b2f61c7b1710fcc6826fc0cc12dca8c59c0fbdadd82c05bb6ebff2028583fcf6"
CONTRIEVER_PATH = ROOT / "artifacts" / "models" / "contriever_msmarco"
CONTRIEVER_SHA256 = "08b88f3a3697877345669405c51a23f53ed90aa2bab441cd7b7b08659925eef8"
RANKINGS = ("lexical", "dense", "fusion")
POLICIES = ("top1", "max_qa5", "consensus10")


def split_rows(path: Path) -> tuple[dict, list[dict]]:
    split = json.loads(path.read_text(encoding="utf-8"))
    rows = [{**row, "split": name} for name in ("calibration", "evaluation")
            for row in split[name]]
    if not rows or len({row["qid"] for row in rows}) != len(rows):
        raise ValueError("Expected distinct frozen questions")
    return split, rows


def pin_models() -> None:
    for path, expected in ((QA_PATH / "pytorch_model.bin", QA_SHA256),
                           (CONTRIEVER_PATH / "pytorch_model.bin", CONTRIEVER_SHA256)):
        if digest(path, "sha256") != expected:
            raise ValueError(f"Pinned model weights changed: {path}")


def retrieve(snapshot: Path, split_path: Path, destination: Path) -> None:
    _, rows = split_rows(split_path)
    completed = 0
    if destination.exists():
        with destination.open("r", encoding="utf-8") as prior:
            for completed, line in enumerate(prior, 1):
                item = json.loads(line)
                if completed > len(rows) or item["qid"] != rows[completed - 1]["qid"]:
                    raise ValueError("Partial retrieval cache differs from frozen split")
                for passage in item["pool"]:
                    if hashlib.sha256(passage["text"].encode("utf-8")).hexdigest() != passage["sha256"]:
                        raise ValueError("Partial retrieval cache contains a changed passage")
    search = TrustedSnapshotSearch(snapshot)
    started = time.perf_counter()
    try:
        with destination.open("a", encoding="utf-8", newline="\n") as out:
            for number in range(completed + 1, len(rows) + 1):
                row = rows[number - 1]
                pool = search.search(row["question"], limit=100)
                out.write(json.dumps({"qid": row["qid"], "pool": pool},
                                     ensure_ascii=False) + "\n")
                if number % 50 == 0:
                    print(f"retrieved {number}/{len(rows)} from 2,681,468 passages "+
                          f"in {time.perf_counter()-started:.0f}s", flush=True)
    finally:
        search.close()


def read_retrievals(path: Path, rows: list[dict]) -> list[dict]:
    by_id = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item["qid"] in by_id:
            raise ValueError("Duplicate retrieved question")
        by_id[item["qid"]] = item["pool"]
    if set(by_id) != {row["qid"] for row in rows}:
        raise ValueError("Retrieval cache differs from frozen question split")
    return [{**row, "pool": by_id[row["qid"]]} for row in rows]


def ranked_top_ten(pool: list[dict], passage_vectors: np.ndarray,
                   query_vector: np.ndarray) -> dict[str, list[dict]]:
    scores = passage_vectors @ query_vector
    dense_order = np.argsort(-scores, kind="stable")
    dense_ranks = np.empty(len(pool), dtype=np.int32)
    dense_ranks[dense_order] = np.arange(1, len(pool) + 1)
    fusion = np.asarray([1.0 / (60 + index + 1) +
                         1.0 / (60 + dense_ranks[index]) for index in range(len(pool))])
    fused_order = np.argsort(-fusion, kind="stable")
    return {
        "lexical": pool[:10],
        "dense": [pool[int(index)] for index in dense_order[:10]],
        "fusion": [pool[int(index)] for index in fused_order[:10]],
    }


def cached_embeddings(texts: list[str], keys: list[str], tokenizer, model,
                      device, embed, directory: Path) -> np.ndarray:
    """Save completed Contriever blocks so an interrupted run can resume."""
    directory.mkdir(parents=True, exist_ok=True)
    key_digest = hashlib.sha256(json.dumps(keys, ensure_ascii=False).encode("utf-8")).hexdigest()
    manifest = directory / "keys.sha256"
    if manifest.exists():
        if manifest.read_text(encoding="utf-8").strip() != key_digest:
            raise ValueError("Embedding cache keys differ from frozen retrievals")
    else:
        manifest.write_text(key_digest + "\n", encoding="utf-8")
    blocks = []
    for start in range(0, len(texts), 2048):
        part = directory / f"part_{start:06d}.npy"
        end = min(start + 2048, len(texts))
        if part.exists():
            block = np.load(part, allow_pickle=False)
            if block.shape[0] != end - start:
                raise ValueError(f"Invalid cached embedding block: {part}")
        else:
            block = embed(texts[start:end], tokenizer, model, device, batch_size=128)
            building = part.with_suffix(".building")
            with building.open("wb") as handle:
                np.save(handle, block)
            os.replace(building, part)
        blocks.append(block)
        print(f"embedded {end}/{len(texts)}", flush=True)
    return np.concatenate(blocks)


def predict(snapshot: Path, split_path: Path, retrieval_path: Path,
            destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite frozen predictions: {destination}")
    metadata = destination.with_suffix(".meta.json")
    if metadata.exists():
        raise FileExistsError(f"Refusing to overwrite timing metadata: {metadata}")
    started = time.perf_counter()
    import torch
    from transformers import AutoModel, AutoTokenizer
    from evaluation.evaluate_trusted_snapshot_v13 import embed

    pin_models()
    _, rows = split_rows(split_path)
    reused_retrieval = retrieval_path.exists()
    retrieve(snapshot, split_path, retrieval_path)
    retrieval_seconds = time.perf_counter() - started
    cases = read_retrievals(retrieval_path, rows)
    unique = {passage["sha256"]: passage["text"]
              for case in cases for passage in case["pool"]}
    print(f"Reranking {len(unique)} unique verified passages", flush=True)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(CONTRIEVER_PATH, local_files_only=True)
    model = AutoModel.from_pretrained(CONTRIEVER_PATH,
                                      local_files_only=True).to(device).eval()
    keys = list(unique)
    vectors = cached_embeddings([unique[key] for key in keys], keys,
                                tokenizer, model, device, embed,
                                destination.parent / "embedding_cache")
    vector_map = dict(zip(keys, vectors))
    question_vectors = embed([case["question"] for case in cases],
                             tokenizer, model, device, batch_size=128)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    for case, vector in zip(cases, question_vectors):
        pool_vectors = np.stack([vector_map[row["sha256"]] for row in case["pool"]])
        case["rankings"] = ranked_top_ten(case["pool"], pool_vectors, vector)
        case.pop("pool")
    rerank_seconds = time.perf_counter() - started - retrieval_seconds
    print("Reranking complete; reading exact source spans", flush=True)
    reader = ReferenceReader(QA_PATH)
    pairs = {}
    for case in cases:
        for ranking in RANKINGS:
            for passage in case["rankings"][ranking]:
                pairs[case["qid"], passage["doc_id"]] = (case["question"], passage)
    flat = list(pairs.values())
    predictions = {}
    qa_cache = destination.parent / "reader_cache"
    qa_cache.mkdir(parents=True, exist_ok=True)
    pair_digest = hashlib.sha256(json.dumps([
        (question, passage["doc_id"], passage["sha256"])
        for question, passage in flat], ensure_ascii=False).encode("utf-8")).hexdigest()
    pair_manifest = qa_cache / "pairs.sha256"
    if pair_manifest.exists():
        if pair_manifest.read_text(encoding="utf-8").strip() != pair_digest:
            raise ValueError("Reader cache pairs differ from frozen rankings")
    else:
        pair_manifest.write_text(pair_digest + "\n", encoding="utf-8")
    for start in range(0, len(flat), 128):
        batch = flat[start:start + 128]
        part = qa_cache / f"part_{start:06d}.json"
        if part.exists():
            saved = json.loads(part.read_text(encoding="utf-8"))
            if len(saved) != len(batch):
                raise ValueError(f"Invalid cached reader block: {part}")
        else:
            extracted = reader.extract_batch([question for question, _ in batch],
                                             [passage["text"] for _, passage in batch],
                                             batch_size=8)
            saved = []
            for (question, passage), result in zip(batch, extracted):
                span = passage["text"][result.start:result.end]
                verified = bool(span and span == result.answer and
                                hashlib.sha256(passage["text"].encode("utf-8")).hexdigest()
                                == passage["sha256"])
                saved.append({"answer": result.answer if verified else "",
                              "qa_score": result.score if verified else 0.0,
                              "span_verified": verified})
            building = part.with_suffix(".building")
            building.write_text(json.dumps(saved, ensure_ascii=False) + "\n",
                                encoding="utf-8")
            os.replace(building, part)
        for (question, passage), item in zip(batch, saved):
            predictions[question, passage["doc_id"]] = item
        if (start // 128) % 10 == 0:
            print(f"read {min(start + 128, len(flat))}/{len(flat)} question/passage pairs",
                  flush=True)
    with destination.open("w", encoding="utf-8", newline="\n") as out:
        for case in cases:
            packed = {"qid": case["qid"], "split": case["split"],
                      "question": case["question"], "rankings": {}}
            for ranking in RANKINGS:
                packed["rankings"][ranking] = [
                    {"doc_id": passage["doc_id"], "sha256": passage["sha256"],
                     **predictions[case["question"], passage["doc_id"]]}
                    for passage in case["rankings"][ranking]]
            out.write(json.dumps(packed, ensure_ascii=False) + "\n")
    metadata.write_text(json.dumps({
        "questions": len(cases), "unique_retrieved_passages": len(unique),
        "unique_reader_pairs": len(flat),
        "retrieval_seconds_in_this_run": round(retrieval_seconds, 2),
        "rerank_seconds": round(rerank_seconds, 2),
        "reader_seconds": round(time.perf_counter() - started - retrieval_seconds
                                - rerank_seconds, 2),
        "retrieval_cache_reused": reused_retrieval,
        "retrieval_cache_sha256": digest(retrieval_path, "sha256"),
        "device": str(device),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen answer-blind predictions: {destination}", flush=True)


def strict_normalized(value: str) -> str:
    return normalized(value)


def answer_match(prediction: str, aliases: list[str], *, substring: bool) -> bool:
    value = strict_normalized(prediction)
    if not value:
        return False
    expected = [strict_normalized(alias) for alias in aliases]
    return any((value == alias if not substring else
                f" {value} " in f" {alias} " or f" {alias} " in f" {value} ")
               for alias in expected if alias)


def outcomes(rows: list[dict], predictions: dict[str, dict],
             answers: dict[str, list[str]], threshold: float) -> dict:
    counts = Counter()
    for row in rows:
        chosen = predictions[row["qid"]]
        emitted = chosen["answer"] if chosen["score"] >= threshold else ""
        if not emitted:
            counts["abstained"] += 1
        elif answer_match(emitted, answers[row["qid"]], substring=False):
            counts["strict_match"] += 1
        else:
            counts["nonmatching_emitted"] += 1
        counts["substring_match"] += answer_match(emitted, answers[row["qid"]],
                                                    substring=True)
    return {"questions": len(rows), **{key: counts[key] for key in
            ("strict_match", "substring_match", "nonmatching_emitted", "abstained")}}


def calibrate(rows: list[dict], predictions: dict[str, dict],
              answers: dict[str, list[str]]) -> tuple[float, dict]:
    budget = math.floor(len(rows) * 0.10)
    cutoffs = {0.0, 1.0, *(predictions[row["qid"]]["score"] for row in rows)}
    cutoffs.add(math.nextafter(max(cutoffs), math.inf))
    options = []
    for cutoff in cutoffs:
        counts = outcomes(rows, predictions, answers, cutoff)
        if counts["nonmatching_emitted"] <= budget:
            options.append((counts["strict_match"], -counts["nonmatching_emitted"],
                            -cutoff, cutoff, counts))
    _, _, _, cutoff, counts = max(options)
    return cutoff, counts


def score(split_path: Path, predictions_path: Path, answers_path: Path,
          archive: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {destination}")
    split, rows = split_rows(split_path)
    if digest(answers_path, "sha256") != split["nq_open_answers_sha256"]:
        raise ValueError("NQ-Open answer file changed")
    if digest(archive, "md5") != BEIR_MD5:
        raise ValueError("BEIR source archive changed")
    by_text: dict[str, list[dict]] = defaultdict(list)
    for line in answers_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            by_text[normalized(item["question"])].append(item)
    answers = {}
    for row in rows:
        matches = by_text[normalized(row["question"])]
        if len(matches) != 1 or not isinstance(matches[0]["answer"], list):
            raise ValueError(f"Unmatched or ambiguous official answer: {row['qid']}")
        answers[row["qid"]] = matches[0]["answer"]
    predictions = {}
    for line in predictions_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item["qid"] in predictions:
            raise ValueError("Duplicate prediction")
        predictions[item["qid"]] = item
    if set(predictions) != {row["qid"] for row in rows}:
        raise ValueError("Predictions differ from frozen split")
    with ZipFile(archive) as source:
        qrel_bytes = source.read("nq/qrels/test.tsv")
    if hashlib.sha256(qrel_bytes).hexdigest() != split["beir_qrels_sha256"]:
        raise ValueError("BEIR qrels changed")
    relevant: dict[str, set[str]] = defaultdict(set)
    for row in csv.DictReader(io.StringIO(qrel_bytes.decode("utf-8")), delimiter="\t"):
        if int(row["score"]) > 0:
            relevant[row["query-id"]].add(row["corpus-id"])
    calibration_rows = split["calibration"]
    evaluation_rows = split["evaluation"]
    options = []
    for rank_index, ranking in enumerate(RANKINGS):
        for policy_index, policy in enumerate(POLICIES):
            chosen = {qid: policy_prediction(item["rankings"][ranking], policy)
                      for qid, item in predictions.items()}
            cutoff, counts = calibrate(calibration_rows, chosen, answers)
            options.append((counts["strict_match"], -counts["nonmatching_emitted"],
                            -rank_index, -policy_index, ranking, policy, cutoff,
                            counts))
    _, _, _, _, ranking, policy, cutoff, calibration = max(options)
    chosen = {qid: policy_prediction(item["rankings"][ranking], policy)
              for qid, item in predictions.items()}
    evaluation = outcomes(evaluation_rows, chosen, answers, cutoff)
    retrieval = {}
    for name, subset in (("calibration", calibration_rows),
                         ("evaluation", evaluation_rows)):
        retrieval[name] = {method: {f"qrel_at_{k}": sum(bool(
            relevant[row["qid"]] & {p["doc_id"] for p in
                                    predictions[row["qid"]]["rankings"][method][:k]})
            for row in subset) for k in (1, 5, 10)} for method in RANKINGS}
    source_failures = sum(not p["span_verified"] for item in predictions.values()
                          for method in RANKINGS for p in item["rankings"][method])
    unverified_emissions = 0
    for row in rows:
        selected = chosen[row["qid"]]
        if selected["answer"] and selected["score"] >= cutoff:
            verified_rows = [p for p in predictions[row["qid"]]["rankings"][ranking]
                             if p["doc_id"] in selected["doc_ids"] and
                             p["span_verified"] and p["answer"] == selected["answer"]]
            if not verified_rows:
                unverified_emissions += 1
    case_path = destination.with_suffix(".cases.jsonl")
    if case_path.exists():
        raise FileExistsError(f"Refusing to overwrite case rows: {case_path}")
    with case_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            selected = chosen[row["qid"]]
            answer = selected["answer"] if selected["score"] >= cutoff else ""
            handle.write(json.dumps({"qid": row["qid"], "split": row["split"],
                "question": row["question"], "answer": answer,
                "aliases": answers[row["qid"]], "selected_doc_ids": selected["doc_ids"],
                "strict_match": answer_match(answer, answers[row["qid"]], substring=False),
                "retrieved_doc_ids": [p["doc_id"] for p in
                                      predictions[row["qid"]]["rankings"][ranking]]},
                ensure_ascii=False) + "\n")
    report = {
        "protocol": "research_protocols/full_corpus_v17.md",
        "split_sha256": digest(split_path, "sha256"),
        "predictions_sha256": digest(predictions_path, "sha256"),
        "cases_sha256": digest(case_path, "sha256"),
        "beir_passages": 2681468,
        "calibration_questions": 200, "evaluation_questions": 500,
        "selected_ranking": ranking, "selected_policy": policy,
        "calibrated_score_cutoff": cutoff,
        "calibration_options": [{"ranking": item[4], "policy": item[5],
                                 "cutoff": item[6], **item[7]} for item in options],
        "calibration": calibration, "evaluation": evaluation,
        "retrieval": retrieval,
        "reader_source_span_verification_failures": source_failures,
        "emitted_unverified_sources": unverified_emissions,
        "gate": {"strict_match_at_least": 200,
                 "nonmatching_emitted_at_most": 50,
                 "emitted_unverified_sources": 0},
        "gate_passed": evaluation["strict_match"] >= 200
                       and evaluation["nonmatching_emitted"] <= 50
                       and unverified_emissions == 0,
        "limitations": ["Public NQ benchmark, not an operational corpus",
                        "Clean QA only; no poisoning attack was run in v17",
                        "Exact string scoring can miss valid paraphrases",
                        "Source hash proves snapshot identity, not factual truth"],
    }
    metadata = predictions_path.with_suffix(".meta.json")
    if metadata.exists():
        report["inference_timing"] = json.loads(metadata.read_text(encoding="utf-8"))
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("predict", "score"))
    parser.add_argument("--snapshot", type=Path,
                        default=ROOT / "data" / "trusted_snapshot_v12")
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--beir-archive", type=Path, required=True)
    parser.add_argument("--answers-file", type=Path,
                        default=LOCAL / "NQ-open.dev.jsonl")
    parser.add_argument("--local", type=Path, default=LOCAL,
                        help="Run-specific ignored cache and prediction directory")
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args()
    if args.phase == "predict":
        args.local.mkdir(parents=True, exist_ok=True)
        predict(args.snapshot, args.split, args.local / "retrievals.jsonl",
                args.local / "predictions.jsonl")
    else:
        report = score(args.split, args.local / "predictions.jsonl", args.answers_file,
                       args.beir_archive, args.output)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
