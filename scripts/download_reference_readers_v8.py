"""Fetch pinned local QA readers used by the v8 research replay.

The selected Natural Questions reader is the default. Pass --include-tiny to
also fetch the smaller development comparison reader. Model weights are
gitignored and checked against their published SHA-256 values.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download


READERS = {
    "roberta_nq": {
        "repo": "nlpconnect/roberta-base-squad2-nq",
        "revision": "8725d9fd10e9288c94fb259b513190b045c54af6",
        "directory": Path("artifacts/vendor/roberta-base-squad2-nq"),
        "weight": "pytorch_model.bin",
        "sha256": "b2f61c7b1710fcc6826fc0cc12dca8c59c0fbdadd82c05bb6ebff2028583fcf6",
    },
    "tinyroberta": {
        "repo": "deepset/tinyroberta-squad2",
        "revision": "75c968b8219e055ce1a89533057b142bae65450f",
        "directory": Path("artifacts/vendor/tinyroberta-squad2"),
        "weight": "model.safetensors",
        "sha256": "39955cbf5f636762a2ef642b718c8c6e172d153c94a13de40b37efd00822cd80",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def download(name: str) -> None:
    reader = READERS[name]
    snapshot_download(
        repo_id=reader["repo"], revision=reader["revision"],
        local_dir=reader["directory"],
        allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json",
                        "special_tokens_map.json", "merges.txt", "vocab.json",
                        reader["weight"]],
    )
    weight = reader["directory"] / reader["weight"]
    if sha256(weight) != reader["sha256"]:
        raise ValueError(f"Model weight hash mismatch: {weight}")
    print(f"Verified {reader['repo']} at {reader['revision']}: {weight}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-tiny", action="store_true")
    args = parser.parse_args()
    download("roberta_nq")
    if args.include_tiny:
        download("tinyroberta")


if __name__ == "__main__":
    main()
