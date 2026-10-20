"""Write two pinned NFCorpus cases for hosted V24/V25 smoke checks."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cases = [json.loads(line) for line in
             (ROOT / "data/v25_nfcorpus_holdout/cases.jsonl").read_text(
                 encoding="utf-8").splitlines()]
    traces = [json.loads(line) for line in
              (ROOT / "results/v25_nfcorpus_holdout.cases.jsonl").read_text(
                  encoding="utf-8").splitlines()]
    if len(cases) != len(traces):
        raise ValueError("Holdout case and trace counts differ")
    selected: dict[str, tuple[dict, dict]] = {}
    for case, trace in zip(cases, traces, strict=True):
        if (case["query_id"], case["candidate_id"]) != (
                trace["query_id"], trace["candidate_id"]):
            raise ValueError("Holdout case and trace order differs")
        if (case["label"] == "poison" and trace["v25_flag"]
                and not trace["v24_flag"] and "attack" not in selected):
            selected["attack"] = case, trace
        if (case["label"] != "poison" and not trace["v25_flag"]
                and not trace["v24_flag"] and "clean" not in selected):
            selected["clean"] = case, trace
    if set(selected) != {"attack", "clean"}:
        raise ValueError("Could not find both smoke cases")
    target = ROOT / "tmp"
    target.mkdir(exist_ok=True)
    for name, (case, trace) in selected.items():
        request = {key: case[key] for key in ("query", "reference", "candidate")}
        (target / f"v25_smoke_{name}.json").write_text(
            json.dumps(request, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"case": name, "query_id": case["query_id"],
                          "family": case["family"],
                          "v24_flag": trace["v24_flag"],
                          "v25_flag": trace["v25_flag"]}))


if __name__ == "__main__":
    main()
