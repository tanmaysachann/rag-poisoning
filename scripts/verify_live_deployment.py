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
    if not all(marker in page for marker in ("lab-case", "lab-ppo-cache-ablation", "lab-unseen-summary")):
        raise ValueError("Production page is missing research lab bindings")
    health = get(base_url, "/api/health")
    if health.get("status") != "operational":
        raise ValueError("Production health endpoint is not operational")
    summary = get(base_url, "/api/research-summary")
    if (summary.get("split") != "validation"
            or summary["summaries"]["ppo_multiseed"]["ppo_successes"] != 31
            or summary["summaries"]["unseen_templates"]["styles"]["qa_header"]["cases"] != 75
            or summary["summaries"]["ppo_head_ablation"]["changed_case_outcomes"] != 0):
        raise ValueError("Production research evidence differs from committed results")
    catalog = get(base_url, "/api/lab/cases")
    if catalog.get("split") != "validation" or len(catalog["cases"]) != 75:
        raise ValueError("Production case catalog differs")
    sealed = get(base_url, "/api/test-summary")
    if (sealed.get("protocol") != "v1" or not sealed.get("available")
            or sealed["summaries"]["clean"]["cases"] != 75):
        raise ValueError("Production frozen-test aggregate failed verification")
    live = post(base_url, "/api/lab/run", {
        "qid": "msmarco-275", "strategy": "stealth", "surface": "accepted_ingest",
    })
    if (live.get("qid") != "msmarco-275" or len(live["documents"]) != 5
            or [row["stage"] for row in live["audit"]] != [
                "retrieval", "integrity", "detection", "provenance", "answer",
            ]):
        raise ValueError("Production live inference trace is incomplete")
    return {
        "status": "verified", "url": base_url, "health": health["status"],
        "validation_cases": len(catalog["cases"]), "frozen_test_cases": 75,
        "live_documents": len(live["documents"]), "live_audit_stages": len(live["audit"]),
        "live_latency_ms": live["latency_ms"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://rag-poisoning.vercel.app")
    args = parser.parse_args()
    print(json.dumps(verify(args.url), indent=2))


if __name__ == "__main__":
    main()
