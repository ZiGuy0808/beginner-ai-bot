#!/usr/bin/env python3
"""Verify that the ONNX model and tokenizer belong together."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def tokenizer_vocab_size(tokenizer_path: Path) -> int:
    data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    model = data.get("model", {})
    vocab = model.get("vocab")
    if isinstance(vocab, dict):
        return len(vocab)
    if isinstance(vocab, list):
        return len(vocab)
    raise ValueError(f"Could not find tokenizer model vocab in {tokenizer_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify ONNX model/tokenizer compatibility.")
    parser.add_argument("--model-dir", default="gpt2-tiny-onnx", help="Folder containing model/tokenizer files")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_path = model_dir / "decoder_model.onnx"
    tokenizer_path = model_dir / "tokenizer.json"

    required = [
        "config.json",
        "decoder_model.onnx",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        raise SystemExit(f"Missing files in {model_dir}: {', '.join(missing)}")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    print("ONNX Runtime:", ort.__version__)
    print("Providers:", ort.get_available_providers())

    print("\nInputs:")
    for item in session.get_inputs():
        print(f"  {item.name}: shape={item.shape}, type={item.type}")

    print("\nOutputs:")
    for item in session.get_outputs():
        print(f"  {item.name}: shape={item.shape}, type={item.type}")

    input_ids = np.array([[0]], dtype=np.int64)
    attention_mask = np.array([[1]], dtype=np.int64)
    outputs = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
    logits = outputs[0]
    model_vocab = int(logits.shape[-1])
    tok_vocab = tokenizer_vocab_size(tokenizer_path)

    print("\nModel logits shape:", logits.shape)
    print("Model vocab size:", model_vocab)
    print("Tokenizer vocab size:", tok_vocab)

    if model_vocab != tok_vocab:
        raise SystemExit("FAIL: tokenizer vocab size does not match model vocab size")

    print("\nPASS: tokenizer and ONNX model vocab sizes match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
