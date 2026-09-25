"""Resume the pinned Qwen 0.5B weight when Hugging Face Xet stalls.

The remaining tokenizer/config files come from the pinned model snapshot.
This script downloads the large weight through the official Hugging Face
resolve endpoint and verifies its repository-linked SHA-256 before use.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
WEIGHT_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
WEIGHT_BYTES = 988097824
URL = f"https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/{REVISION}/model.safetensors"
TARGET = ROOT / "artifacts/models/qwen2_5_0_5b/model.safetensors"
SUPPORT_FILES = ("config.json", "generation_config.json", "merges.txt",
                 "tokenizer.json", "tokenizer_config.json", "vocab.json")


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    for filename in SUPPORT_FILES:
        path = TARGET.parent / filename
        if path.is_file() and path.stat().st_size:
            continue
        url = f"https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/{REVISION}/{filename}"
        with session.get(url, timeout=(20, 90)) as response:
            response.raise_for_status()
            temporary = path.with_name(path.name + ".part")
            temporary.write_bytes(response.content)
            temporary.replace(path)
    if TARGET.is_file() and _sha256(TARGET) == WEIGHT_SHA256:
        print(f"Verified existing {TARGET}")
        return
    part = TARGET.with_suffix(".safetensors.part")
    for attempt in range(5):
        offset = part.stat().st_size if part.exists() else 0
        if offset == WEIGHT_BYTES:
            break
        if offset > WEIGHT_BYTES:
            raise ValueError("Partial file exceeds the pinned weight size")
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with session.get(URL, headers=headers, stream=True, timeout=(20, 90)) as response:
                response.raise_for_status()
                if offset and (response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                    raise RuntimeError("Server did not honor the resume offset")
                with part.open("ab") as handle:
                    last_report = offset
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        if handle.tell() - last_report >= 64 * 1024 * 1024:
                            last_report = handle.tell()
                            print(f"Downloaded {handle.tell() / WEIGHT_BYTES:.1%}", flush=True)
        except requests.RequestException as error:
            print(f"Transfer interrupted at {part.stat().st_size if part.exists() else 0} bytes: {error}", flush=True)
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 8))
    if part.stat().st_size != WEIGHT_BYTES:
        raise RuntimeError(f"Incomplete model weight: {part.stat().st_size}/{WEIGHT_BYTES} bytes")
    digest = _sha256(part)
    if digest != WEIGHT_SHA256:
        raise RuntimeError(f"Model weight SHA-256 mismatch: {digest}")
    part.replace(TARGET)
    print(f"Verified {TARGET} ({WEIGHT_BYTES} bytes, SHA-256 {digest})")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
