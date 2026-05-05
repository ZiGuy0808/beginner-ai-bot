#!/usr/bin/env python3
"""Download a complete tiny ONNX model bundle safely.

This script does not bypass sandbox rules. It only performs normal HTTPS downloads
for known model files into a local folder.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_REPO = "https://huggingface.co/fxmarty/gpt2-tiny-onnx/resolve/main"
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, force: bool = False) -> None:
    if dest.exists() and not force:
        print(f"[skip] {dest} already exists")
        return

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"[download] {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not download {url}: {exc}") from exc

    tmp.write_bytes(data)
    tmp.replace(dest)
    print(f"[ok] {dest} ({dest.stat().st_size:,} bytes, sha256={sha256_file(dest)[:16]}...)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download tiny ONNX model files.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Base resolve URL for model files")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory")
    parser.add_argument("--force", action="store_true", help="Redownload existing files")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        url = f"{args.repo.rstrip('/')}/{name}"
        download(url, out / name, force=args.force)

    print("\nDone. Next run:")
    print("  python verify_model.py")
    print("  python run_generate.py \"Hello world\" --max-new-tokens 20")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
