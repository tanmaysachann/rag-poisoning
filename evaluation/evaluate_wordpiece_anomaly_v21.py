"""Measure WordPiece fragmentation as a cheap typo/noise signal."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from detect.research_detector import threshold_for_fpr
from detect.train_multifamily_v20 import read_verified_rows
from evaluation.evaluate_multifamily_v20 import summarize

WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def score(text: str, tokenizer) -> dict[str, float]:
    tokens = tokenizer.tokenize(text[:3000])
    words = WORD_RE.findall(text[:3000])
    if not tokens or not words:
        return {"fertility": 0.0, "continuation": 0.0, "one_char_piece": 0.0}
    continuations = [token for token in tokens if token.startswith("##")]
    one_char = [token for token in tokens if len(token.removeprefix("##")) == 1]
    return {"fertility": len(tokens) / len(words),
            "continuation": len(continuations) / len(tokens),
            "one_char_piece": len(one_char) / len(tokens)}


def main() -> None:
    data = Path("data/multifamily_v20")
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibration = read_verified_rows(data, "calibration", manifest)
    evaluation = read_verified_rows(data, "evaluation", manifest)
    tokenizer = AutoTokenizer.from_pretrained("artifacts/multifamily_v20", local_files_only=True)
    cal = [score(row["candidate"], tokenizer) for row in calibration]
    dev = [score(row["candidate"], tokenizer) for row in evaluation]
    clean = np.asarray([row["label"] != "poison" for row in calibration])
    report = {"status": "exploratory development", "signals": {},
              "data_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "tokenizer": "MiniLM WordPiece from the trained V20 model"}
    best = None
    best_merit = -1.0
    for name in ("fertility", "continuation", "one_char_piece"):
        cal_scores = np.asarray([item[name] for item in cal])
        eval_scores = np.asarray([item[name] for item in dev])
        cutoff = threshold_for_fpr(cal_scores[clean], 0.01)
        calibrated = summarize(calibration, cal_scores, cutoff)
        merit = float(np.mean([row["recall"] for row in calibrated["families"].values()]))
        if merit > best_merit:
            best, best_merit = name, merit
        report["signals"][name] = {
            "calibration_at_1pct": calibrated,
            "evaluation_at_1pct": summarize(evaluation, eval_scores, cutoff),
        }
    report["selected_on_calibration"] = best
    report["selected_calibration_macro_recall"] = best_merit
    output = Path("results/wordpiece_anomaly_v21.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": best, "evaluation": report["signals"][best]["evaluation_at_1pct"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
