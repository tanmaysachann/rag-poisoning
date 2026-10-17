"""Smoke-check the public research lab after a production deployment."""

from __future__ import annotations

import argparse
import json
from urllib.request import Request, urlopen


def get(base_url: str, path: str) -> dict | str:
    with urlopen(f"{base_url}{path}", timeout=90) as response:
        if response.status != 200:
            raise ValueError(f"GET {path}: HTTP {response.status}")
        body = response.read().decode("utf-8")
    return json.loads(body) if path.startswith("/api/") else body


def post(base_url: str, path: str, payload: dict) -> dict:
    request = Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=90) as response:
        if response.status != 200:
            raise ValueError(f"POST {path}: HTTP {response.status}")
        return json.load(response)


def verify(base_url: str) -> dict:
    base_url = base_url.rstrip("/")
    page = get(base_url, "/")
    if not all(marker in page for marker in ("lab-case", "lab-payload-2", "lab-payload-3", "lab-ppo-cache-ablation",
                                                "lab-unseen-summary", "lab-budget-summary",
                                                "benchmark label, not a website version")):
        raise ValueError("Production page is missing research lab bindings")
    health = get(base_url, "/api/health")
    if health.get("status") != "operational":
        raise ValueError("Production health endpoint is not operational")
    summary = get(base_url, "/api/research-summary")
    if (summary.get("split") != "validation"
            or summary["summaries"]["ppo_multiseed"]["ppo_successes"] != 31
            or summary["summaries"]["unseen_templates"]["styles"]["qa_header"]["cases"] != 75
            or summary["summaries"]["poison_budget"]["budgets"]["3"]["defended_attack_successes"] != 9
            or summary["summaries"]["ppo_head_ablation"]["changed_case_outcomes"] != 0):
        raise ValueError("Production research evidence differs from committed results")
    catalog = get(base_url, "/api/lab/cases")
    if (catalog.get("split") != "validation" or catalog.get("benchmark") != "v3"
            or len(catalog["cases"]) != 150):
        raise ValueError("Production case catalog differs")
    generalization = get(base_url, "/api/generalization-summary")
    if (generalization["verification"].get("status") != "verified"
            or generalization["verification"].get("queries") != 150
            or generalization["summary"]["paired_attack_families"]["faq_record"]["attack_success_after_detector"] != 65):
        raise ValueError("Production frozen held-out generalization evidence failed verification")
    sealed = get(base_url, "/api/test-summary")
    if (sealed.get("protocol") != "v1" or not sealed.get("available")
            or sealed["summaries"]["clean"]["cases"] != 75):
        raise ValueError("Production frozen-test aggregate failed verification")
    qid = catalog["default_qid"]
    live = post(base_url, "/api/lab/run", {
        "qid": qid, "strategy": "stealth", "surface": "accepted_ingest",
    })
    if (live.get("qid") != qid or len(live["documents"]) != 5
            or [row["stage"] for row in live["audit"]] != [
                "retrieval", "integrity", "detection", "provenance", "answer",
            ]):
        raise ValueError("Production live inference trace is incomplete")
    replicated = post(base_url, "/api/lab/run", {
        "qid": qid, "strategy": "stealth",
        "surface": "accepted_ingest", "attack_budget": 3,
    })
    if (replicated.get("attack_budget") != 3 or len(replicated["attack_doc_ids"]) != 3
            or replicated.get("attack_payloads_distinct") != 1
            or replicated["attack_retrieved_count"] != sum(row["is_attack"] for row in replicated["documents"])):
        raise ValueError("Production multi-passage attack control is incomplete")
    base_text = next(row["stealth_text"] for row in catalog["cases"] if row["qid"] == qid)
    varied = post(base_url, "/api/lab/run", {
        "qid": qid, "strategy": "stealth", "surface": "accepted_ingest",
        "attack_budget": 3, "attack_text": base_text,
        "additional_attack_texts": [base_text + " Variant two.", base_text + " Variant three."],
    })
    retrieved_texts = {row["text"] for row in varied["documents"] if row["is_attack"]}
    if (varied.get("attack_payloads_distinct") != 3
            or not retrieved_texts.intersection({base_text + " Variant two.", base_text + " Variant three."})):
        raise ValueError("Production varied-passage attack control is incomplete")
    return {
        "status": "verified", "url": base_url, "health": health["status"],
        "validation_cases": len(catalog["cases"]), "frozen_test_cases": 75,
        "heldout_generalization_queries": generalization["verification"]["queries"],
        "live_documents": len(live["documents"]), "live_audit_stages": len(live["audit"]),
        "multi_attack_budget": replicated["attack_budget"],
        "varied_attack_payloads": varied["attack_payloads_distinct"],
        "live_latency_ms": live["latency_ms"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://rag-poisoning.vercel.app")
    args = parser.parse_args()
    print(json.dumps(verify(args.url), indent=2))


if __name__ == "__main__":
    main()
