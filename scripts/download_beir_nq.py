"""Download official BEIR NQ archive to temp storage with resumable ranges.

The full corpus is needed only to extract a bounded set of original passages.
The official archive MD5 is from the BEIR dataset table.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip"
SIZE = 498307926
MD5 = "d4d3d2e48787a744b6f6e691ff534307"


def _download_part(index: int, start: int, end: int, directory: Path) -> Path:
    path = directory / f"part-{index:02d}"
    expected = end - start + 1
    for attempt in range(3):
        have = path.stat().st_size if path.exists() else 0
        if have == expected:
            return path
        if have > expected:
            raise ValueError(f"Oversized range part: {path}")
        offset = start + have
        try:
            with requests.get(URL, headers={"Range": f"bytes={offset}-{end}"},
                              stream=True, timeout=90) as response:
                response.raise_for_status()
                if response.status_code != 206 or not response.headers.get(
                    "Content-Range", ""
                ).startswith(f"bytes {offset}-{end}/"):
                    raise ValueError(f"Server returned an unexpected range for {index}")
                with path.open("ab") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if path.stat().st_size == expected:
                print(f"range {index} complete ({expected} bytes)", flush=True)
                return path
        except (requests.RequestException, OSError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Range {index} incomplete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(tempfile.gettempdir()) / "sentinel-rag-beir-nq.zip")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and hashlib.md5(args.output.read_bytes()).hexdigest() == MD5:
        print(f"Already verified: {args.output}")
        return
    part_dir = args.output.parent / "sentinel-rag-beir-nq.parts"
    part_dir.mkdir(exist_ok=True)
    width = (SIZE + args.workers - 1) // args.workers
    ranges = [(i, i * width, min((i + 1) * width, SIZE) - 1)
              for i in range(args.workers) if i * width < SIZE]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_download_part, i, start, end, part_dir)
                   for i, start, end in ranges]
        for future in as_completed(futures):
            future.result()
    temporary = args.output.with_suffix(".assembled.tmp")
    digest = hashlib.md5()
    with temporary.open("wb") as result:
        for index, _, _ in ranges:
            with (part_dir / f"part-{index:02d}").open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    result.write(chunk)
                    digest.update(chunk)
    if digest.hexdigest() != MD5 or temporary.stat().st_size != SIZE:
        raise ValueError("Downloaded BEIR NQ archive failed size/MD5 verification")
    os.replace(temporary, args.output)
    for index, _, _ in ranges:
        (part_dir / f"part-{index:02d}").unlink()
    part_dir.rmdir()
    print(f"Verified {args.output}: {SIZE} bytes, MD5 {MD5}")


if __name__ == "__main__":
    main()
