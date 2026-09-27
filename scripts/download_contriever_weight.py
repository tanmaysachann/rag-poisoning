"""Resume and verify the pinned official Contriever-msmarco model files."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
REVISION = "abe8c1493371369031bcb1e02acb754cf4e162fa"
WEIGHT_SHA256 = "08b88f3a3697877345669405c51a23f53ed90aa2bab441cd7b7b08659925eef8"
WEIGHT_BYTES = 438007537
REPO_URL = f"https://huggingface.co/facebook/contriever-msmarco/resolve/{REVISION}"
TARGET = ROOT / "artifacts/models/contriever_msmarco/pytorch_model.bin"
SUPPORT_FILES = ("config.json", "special_tokens_map.json", "tokenizer.json",
                 "tokenizer_config.json", "vocab.txt")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    for filename in SUPPORT_FILES:
        path = TARGET.parent / filename
        if path.is_file() and path.stat().st_size:
            continue
        with session.get(f"{REPO_URL}/{filename}", timeout=(20, 120)) as response:
            response.raise_for_status()
            temporary = path.with_name(path.name + ".part")
            temporary.write_bytes(response.content)
            temporary.replace(path)
    if TARGET.is_file() and _sha256(TARGET) == WEIGHT_SHA256:
        print(f"Verified existing {TARGET}")
        return
    part = TARGET.with_name(TARGET.name + ".part")
    for attempt in range(8):
        offset = part.stat().st_size if part.exists() else 0
        if offset == WEIGHT_BYTES:
            break
        if offset > WEIGHT_BYTES:
            raise ValueError("Partial file exceeds the pinned weight size")
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with session.get(f"{REPO_URL}/pytorch_model.bin", headers=headers,
                             stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                if offset and (response.status_code != 206 or not response.headers.get(
                        "Content-Range", "").startswith(f"bytes {offset}-")):
                    raise RuntimeError("Server did not honor the resume offset")
                with part.open("ab") as handle:
                    last_report = offset
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        if handle.tell() - last_report >= 32 * 1024 * 1024:
                            last_report = handle.tell()
                            print(f"Downloaded {handle.tell() / WEIGHT_BYTES:.1%}", flush=True)
        except requests.RequestException as error:
            print(f"Transfer interrupted at {part.stat().st_size if part.exists() else 0} bytes: {error}",
                  flush=True)
            if attempt == 7:
                raise
            time.sleep(min(2 ** attempt, 16))
    if not part.is_file() or part.stat().st_size != WEIGHT_BYTES:
        raise RuntimeError(f"Incomplete model weight: {part.stat().st_size if part.exists() else 0}/{WEIGHT_BYTES}")
    digest = _sha256(part)
    if digest != WEIGHT_SHA256:
        raise RuntimeError(f"Model weight SHA-256 mismatch: {digest}")
    part.replace(TARGET)
    print(f"Verified {TARGET} ({WEIGHT_BYTES} bytes, SHA-256 {digest})")


if __name__ == "__main__":
    main()
