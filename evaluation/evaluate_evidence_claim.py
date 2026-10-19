"""Measure the pinned evidence screen on the existing v3 validation replay.

This is a source-authorization evaluation, not an estimate of semantic poison
detection. The optional factual model is reported as a diagnostic only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from detect.evidence_claim import ChangedClaimModel, EvidenceScreen, QuestionClaimModel
from security.integrity import load_manifest, write_manifest


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "data" / "benchmark_v3" / "validation"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-diagnostic", action="store_true")
    parser.add_argument("--qa-diagnostic", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "evidence_claim_screen_validation.json")
    args = parser.parse_args()
    benchmark = json.loads((ROOT / "data" / "benchmark_v3" / "manifest.json")
                           .read_text(encoding="utf-8"))
    corpus_path = VALIDATION / "corpus.jsonl"
    expected_digest = benchmark["file_sha256"]["validation/corpus.jsonl"]
    if hashlib.sha256(corpus_path.read_bytes()).hexdigest() != expected_digest:
        raise ValueError("The preattack evidence corpus differs from its pinned version")
    clean = _jsonl(corpus_path)
    clean_by_id = {int(doc["doc_id"]): str(doc["text"]) for doc in clean}
    questions = {row["qid"]: row["question"]
                 for row in _jsonl(VALIDATION / "queries.jsonl")}
    stealth_rows = []
    with tempfile.TemporaryDirectory(prefix="sentinel-evidence-eval-") as temporary:
        manifest_path = Path(temporary) / "snapshot.json"
        write_manifest(manifest_path, clean, expected_digest)
        snapshot = load_manifest(manifest_path)
        source = SimpleNamespace(corpus_digest=expected_digest, documents=clean)
        screen = EvidenceScreen.from_retriever(snapshot, source)
        clean_decisions = [screen.assess(doc) for doc in clean]
        attack_counts = {}
        changed_pairs = []
        for strategy in ("greedy", "stealth"):
            attacks = _jsonl(ROOT / "results" / f"v3_attack_validation_{strategy}.jsonl")
            counts = Counter()
            for attack in attacks:
                doc = {"doc_id": attack["attack_doc_id"], "text": attack["text"]}
                decision = screen.assess(doc)
                counts[decision["snapshot_status"]] += 1
                counts["quarantined"] += decision["decision"] == "quarantine"
                if decision["changed_claim_pair"]:
                    changed_pairs.append(decision["changed_claim_pair"])
                if strategy == "stealth" and int(doc["doc_id"]) in clean_by_id:
                    stealth_rows.append({
                        "question": questions[attack["qid"]],
                        "source_text": clean_by_id[int(doc["doc_id"])],
                        "candidate_text": attack["text"],
                    })
            attack_counts[strategy] = {**counts, "total": len(attacks)}
    report = {
        "task": "independent pinned-source screening before answer selection",
        "split": "previously_inspected_v3_validation",
        "pinned_corpus_sha256": expected_digest,
        "clean": {"total": len(clean_decisions),
                  "quarantined": sum(row["decision"] == "quarantine"
                                     for row in clean_decisions)},
        "attacks": attack_counts,
        "changed_sentence_pairs": len(changed_pairs),
        "model_diagnostic": None,
        "qa_diagnostic": None,
        "limitations": [
            "The attacker cannot change the pinned preattack corpus or its manifest.",
            "This recognizes unauthorized source changes, not factual falsity in an approved source.",
            "Validation questions and attack families were previously inspected.",
            "A new legitimate document is also unreviewed until an operator approves it.",
        ],
    }
    if args.model_diagnostic:
        model = ChangedClaimModel(ROOT / "artifacts" / "vendor" / "albert-base-vitaminc")
        scores = model.score(changed_pairs)
        report["model_diagnostic"] = {
            "model": "tals/albert-base-vitaminc",
            "count": len(scores),
            "mean_refutation_probability": sum(scores) / len(scores) if scores else None,
            "at_least_0_5": sum(score >= 0.5 for score in scores),
            "at_least_0_9": sum(score >= 0.9 for score in scores),
            "note": "Uncalibrated sentence-pair diagnostic; no model-based quarantine or accuracy claim.",
        }
    if args.qa_diagnostic:
        qa = QuestionClaimModel(ROOT / "artifacts" / "vendor" / "roberta-base-squad2-nq")
        comparisons = qa.compare(stealth_rows)
        report["qa_diagnostic"] = {
            "model": "nlpconnect/roberta-base-squad2-nq",
            "count": len(comparisons),
            "valid_spans": sum(row["status"] == "compared" for row in comparisons),
            "answer_disagreements": sum(row["claim_disagrees"] for row in comparisons),
            "disagreements_by_minimum_confidence": {
                str(cutoff): sum(row["claim_disagrees"]
                                 and min(row["source_confidence"],
                                         row["candidate_confidence"]) >= cutoff
                                 for row in comparisons)
                for cutoff in (0.05, 0.1, 0.25, 0.5)
            },
            "note": "Independent source is matched by document identity. No legitimate revisions or new-source cases establish a deployable false-flag rate.",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
