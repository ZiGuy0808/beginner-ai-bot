#!/usr/bin/env python3
"""Run a simple greedy generation loop with a tiny ONNX decoder model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def softmax_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    logits = logits.astype(np.float64)
    logits = logits / max(temperature, 1e-6)
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    return probs / np.sum(probs)


def pick_next_token(logits: np.ndarray, temperature: float = 0.0, top_k: int = 0) -> int:
    last = logits[0, -1]

    if temperature <= 0:
        return int(np.argmax(last))

    if top_k and top_k > 0:
        k = min(top_k, last.shape[-1])
        idx = np.argpartition(last, -k)[-k:]
        filtered = last[idx]
        probs = softmax_temperature(filtered, temperature)
        return int(np.random.choice(idx, p=probs))

    probs = softmax_temperature(last, temperature)
    return int(np.random.choice(np.arange(last.shape[-1]), p=probs))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate text with a tiny ONNX model.")
    parser.add_argument("prompt", nargs="?", default="Hello world", help="Prompt text")
    parser.add_argument("--model-dir", default="gpt2-tiny-onnx", help="Folder containing model/tokenizer files")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 means greedy argmax")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    np.random.seed(args.seed)

    model_dir = Path(args.model_dir)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    session = ort.InferenceSession(str(model_dir / "decoder_model.onnx"), providers=["CPUExecutionProvider"])

    encoded = tokenizer.encode(args.prompt)
    input_ids = np.array([encoded.ids], dtype=np.int64)
    if input_ids.size == 0:
        input_ids = np.array([[0]], dtype=np.int64)

    generated = input_ids.tolist()[0]
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    for _ in range(args.max_new_tokens):
        current = np.array([generated], dtype=np.int64)
        mask = np.ones_like(current, dtype=np.int64)
        outputs = session.run(None, {"input_ids": current, "attention_mask": mask})
        next_id = pick_next_token(outputs[0], temperature=args.temperature, top_k=args.top_k)
        generated.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break

    text = tokenizer.decode(generated)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
