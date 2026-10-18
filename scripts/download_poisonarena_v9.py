"""Download six independently generated PoisonArena attack files at a pinned commit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests


REPOSITORY = "yxf203/PoisonArena"
COMMIT = "001da5aefa1779f3d2cd1495e0a2d9d614a2fa81"
FILES = {
    "nq_retrievals": "beir_results/nq-contriever-new.json",
    "nq_arena": "arena/data/arena_data/nq-arena.json",
    "nq_queries": "datasets/used_nq/our_queries.jsonl",
    "nq_qrels": "datasets/used_nq/qrels/out_test.tsv",
    "advdecoding": "arena/data/attacker_data/advdecoding/AdvDec5-dcorpus-dot-ans1.json",
    "content_poison": "arena/data/attacker_data/content_poison/cpoison5-inco-ans1.json",
    "corpus_poison": "arena/data/attacker_data/corpus_poison/corpus-dcorpus-answer-ans1.json",
    "garag": "arena/data/attacker_data/garag/GARAG5-ans.json",
    "gaslite": "arena/data/attacker_data/gaslite/gaslite-dcorpus-ans1.json",
    "poisonedrag_black": "arena/data/attacker_data/poisonedrag_black/P5-black-suffix-dcorpus-ans1.json",
}


def main() -> None:
    directory = Path("data/poisonarena_v9")
    directory.mkdir(parents=True, exist_ok=True)
    tree_url = f"https://api.github.com/repos/{REPOSITORY}/git/trees/{COMMIT}?recursive=1"
    response = requests.get(tree_url, timeout=30)
    response.raise_for_status()
    tree = {entry["path"]: entry for entry in response.json()["tree"]}
    manifest = {"repository": REPOSITORY, "commit": COMMIT, "files": {}}
    for family, path in FILES.items():
        if path not in tree or tree[path]["type"] != "blob":
            raise ValueError(f"Pinned file missing: {path}")
        destination = directory / f"{family}.json"
        if destination.exists():
            data = destination.read_bytes()
        else:
            url = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/{path}"
            result = requests.get(url, timeout=180)
            result.raise_for_status()
            data = result.content
        git_hash = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\x00" + data).hexdigest()
        if git_hash != tree[path]["sha"]:
            raise ValueError(f"Git blob hash mismatch: {path}")
        if path.endswith(".json"):
            json.loads(data)
        destination.write_bytes(data)
        manifest["files"][family] = {"path": path, "git_blob_sha1": git_hash,
                                     "sha256": hashlib.sha256(data).hexdigest(),
                                     "bytes": len(data)}
        print(f"{family}: {len(data)} bytes", flush=True)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                               encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
