"""Freeze V24's learned calibration and PPO-tested containment threshold."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace-existing", action="store_true",
                        help="Refresh this script's V24 bundle after evidence-only report changes")
    args = parser.parse_args()
    source = Path("artifacts/product_v22_calibration.joblib")
    output = Path("artifacts/product_v24_calibration.joblib")
    if output.exists() and not args.replace_existing:
        raise FileExistsError(f"Refusing to overwrite {output}")
    if output.exists():
        existing = joblib.load(output)
        if (existing.get("schema") != "product-v24-calibration" or
                existing.get("parent_v22_calibration_sha256") != sha(source)):
            raise ValueError("Refusing to replace an unrelated artifact")
    artifact = joblib.load(source)
    report_path = Path("results/snapshot_containment_v23.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["snapshot_manifest_sha256"] != artifact["snapshot_manifest_sha256"]:
        raise ValueError("Containment evaluated a different source snapshot")
    if report["calibration_clean_flags"] > 2:
        raise ValueError("Containment exceeds 1% clean calibration FPR")
    if report["threshold"] != 0.9:
        raise ValueError("Unexpected containment threshold")
    artifact = dict(artifact)
    artifact["schema"] = "product-v24-calibration"
    artifact["snapshot_threshold"] = report["threshold"]
    artifact["containment_report_sha256"] = sha(report_path)
    artifact["parent_v22_calibration_sha256"] = sha(source)
    joblib.dump(artifact, output)
    print(json.dumps({
        "artifact": str(output), "sha256": sha(output),
        "snapshot_threshold": artifact["snapshot_threshold"],
    }, indent=2))


if __name__ == "__main__":
    main()
