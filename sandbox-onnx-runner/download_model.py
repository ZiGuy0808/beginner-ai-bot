#!/usr/bin/env python3
"""Download a complete tiny ONNX model bundle safely.

This script does not bypass sandbox rules. It only performs normal HTTPS downloads
for known model files into a local folder. It avoids browser copy/paste problems
with large JSON/text files by fetching raw bytes directly.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL_ID = "fxmarty/gpt2-tiny-onnx"
DEFAULT_OUT = Path("gpt2-tiny-onnx")

FILES = [
    "config.json",
    "decoder_model.onnx",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]

# Minimum byte sizes catch failed HTML/error-page downloads and truncated manual copies.
MIN_BYTES = {
    "config.json": 100,
    "decoder_model.onnx": 900_000,
    "merges.txt": 1_000,
    "special_tokens_map.json": 20,
    "tokenizer.json": 20_000,
    "tokenizer_config.json": 20,
    "vocab.json": 10_000,
}


def candidate_urls(model_id: str, filename: str) -> list[str]:
    model_id = model_id.strip("/")
    return [
        f"https://huggingface.co/{model_id}/resolve/main/{filename}?download=true",
        f"https://huggingface.co/{model_id}/resolve/main/{filename}",
        f"https://cdn-lfs.huggingface.co/repos/{model_id}/{filename}",  # usually fails, but harmless fallback
    ]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def looks_like_html(data: bytes) -> bool:
    head = data[:300].lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<title>" in head


def download_one(url: str, timeout: int) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "sandbox-onnx-runner/1.0",
            "Accept": "application/octet-stream,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def download_file(model_id: str, filename: str, dest: Path, force: bool, retries: int, timeout: int) -> None:
    if dest.exists() and not force:
        size = dest.stat().st_size
        min_size = MIN_BYTES.get(filename, 1)
        if size >= min_size:
            print(f"[skip] {dest} already exists ({size:,} bytes)")
            return
        print(f"[redo] {dest} exists but is too small ({size:,} bytes < {min_size:,})")

    errors: list[str] = []
    for url in candidate_urls(model_id, filename):
        for attempt in range(1, retries + 1):
            print(f"[download] {filename} attempt {attempt}/{retries}")
            print(f"           {url}")
            try:
                data = download_one(url, timeout=timeout)
                min_size = MIN_BYTES.get(filename, 1)
                if len(data) < min_size:
                    raise RuntimeError(f"download too small: {len(data):,} bytes < {min_size:,}")
                if looks_like_html(data):
                    raise RuntimeError("download looks like an HTML error/login page, not the raw file")

                tmp = dest.with_suffix(dest.suffix + ".tmp")
                tmp.write_bytes(data)
                tmp.replace(dest)
                print(f"[ok] {dest} ({dest.stat().st_size:,} bytes, sha256={sha256_file(dest)[:16]}...)")
                return
            except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
                errors.append(f"{url}: {exc}")
                time.sleep(1)

    joined = "\n  - ".join(errors[-6:])
    raise RuntimeError(
        f"Could not download {filename}. Last errors:\n  - {joined}\n\n"
        "If the sandbox blocks outbound downloads, manually upload the files into the model folder."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Download tiny ONNX model files without browser copy/paste.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id, e.g. fxmarty/gpt2-tiny-onnx")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory")
    parser.add_argument("--force", action="store_true", help="Redownload existing files")
    parser.add_argument("--retries", type=int, default=3, help="Attempts per URL")
    parser.add_argument("--timeout", type=int, default=90, help="HTTP timeout in seconds")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model_id}")
    print(f"Output: {out.resolve()}\n")

    try:
        for name in FILES:
            download_file(args.model_id, name, out / name, args.force, args.retries, args.timeout)
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print("\nDone. Next run:")
    print("  python verify_model.py")
    print("  python run_generate.py \"Hello world\" --max-new-tokens 20")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
