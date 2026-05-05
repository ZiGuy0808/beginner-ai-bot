# Sandbox ONNX Runner

Safe bootstrap scripts for running a tiny ONNX GPT-2-style model inside a restricted Linux sandbox.

This repo does **not** bypass or escape the sandbox. It only downloads normal model assets, verifies that the tokenizer matches the ONNX model, and runs text generation using allowed local Python execution.

## What this fixes

If a sandbox already has `onnxruntime` installed but only has `decoder_model.onnx`, it still cannot produce readable text without the matching tokenizer files.

This runner downloads the complete tiny model bundle from Hugging Face:

- `config.json`
- `decoder_model.onnx`
- `merges.txt`
- `special_tokens_map.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.json`

Default model:

```text
fxmarty/gpt2-tiny-onnx
```

This model is useful for testing the full pipeline:

```text
text -> tokenizer -> ONNX Runtime -> logits -> token ids -> decoded text
```

It is a tiny/random test model, not a smart chatbot.

## Bob's commands inside the sandbox

```bash
git clone https://github.com/ZiGuy0808/beginner-ai-bot.git
cd beginner-ai-bot/sandbox-onnx-runner
python download_model.py
python verify_model.py
python run_generate.py "Hello world" --max-new-tokens 20
```

## Required Python packages

The sandbox needs:

```text
onnxruntime
numpy
tokenizers
```

If the sandbox has no pip internet access, install wheels offline from a wheelhouse:

```bash
python -m pip install --no-index --find-links=/home/oai/share/wheelhouse onnxruntime numpy tokenizers
```

If `onnxruntime` is already installed, only `tokenizers` may be missing.

## If downloads are blocked

If the sandbox cannot reach Hugging Face, manually place the model files in:

```text
sandbox-onnx-runner/gpt2-tiny-onnx/
```

Then run:

```bash
python verify_model.py
python run_generate.py "Hello" --max-new-tokens 20
```

## Safety boundary

This project intentionally does not attempt to gain extra permissions, break out of a container, access host files, or bypass sandbox restrictions. It only works through normal file downloads and local model execution.
