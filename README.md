# ner-field-guide

Companion repo for the article **“The Definitive Guide to NER in 2026.”**  
Each script demonstrates a NER workflow ranging from zero-shot inference to LLM-assisted labeling.

## 1. Setup

```bash
uv sync
cp .env.example .env  # Required for GPT-4o-powered scripts
```

- Use Python 3.11+ and [uv](https://github.com/astral-sh/uv) for environment management.
- Fill `OPENAI_API_KEY` in `.env` with an API key from [platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys).
- If you proxy requests (Azure/OpenAI-compatible gateways) update `OPENAI_BASE_URL` in `.env`.
- Scripts automatically load `.env` via `python-dotenv`.

## 2. Scripts overview

| Script | Command | What it shows | Notes |
| --- | --- | --- | --- |
| 01 – GLiNER quickstart | `uv run python scripts/01_gliner_quickstart.py` | Minimal zero-shot inference with GLiNER | Purely local, no API keys needed. |
| 02 – ONNX export & INT8 | `uv run python scripts/02_onnx_export.py` | Exports GLiNER to FP32 ONNX and quantizes to INT8 | Outputs live under `artifacts/`. |
| 03 – LLM teacher pipeline | `uv run python scripts/03_llm_teacher_pipeline.py` | Labels unlabeled snippets with GPT-4o (JSON) and produces `teacher_dataset.jsonl` | Falls back to GLiNER-generated pseudo-labels if no API key. |
| 04 – Benchmark | `uv run python scripts/04_benchmark.py` | Side-by-side report comparing GLiNER, GPT teacher labeling, and Instructor extraction | Writes Markdown + JSON reports to `artifacts/`. |
| 05 – Structured extraction | `uv run python scripts/05_structured_extraction.py` | Pydantic-typed extraction via Instructor + GPT-4o | Uses GLiNER fallback if GPT-4o unavailable. |

> All artifacts (ONNX files, labeled datasets, etc.) are written to the `artifacts/` directory which is created on demand.

Or run everything in one go:

```bash
uv run python -m scripts.run_all
```

Add `--ignore-errors` to keep going even if one demo fails.

## 3. Script details

### 01 – GLiNER quickstart

Demonstrates how to load `urchade/gliner_medium-v2.1` and predict a few simple entity types. Outputs labeled entities directly to stdout.

### 02 – ONNX export & INT8 quantization

1. Loads the same GLiNER checkpoint.
2. Exports an FP32 ONNX graph to `artifacts/gliner_medium.onnx`.
3. Runs dynamic INT8 quantization (via `onnxruntime.quantization`) producing `artifacts/gliner_medium_int8.onnx`.
4. Prints the resulting model sizes so you can reason about deployment trade-offs.

### 03 – LLM teacher pipeline

- Reads three unlabeled snippets and asks GPT-4o to return a JSON payload with entities restricted to `["person", "organization", "date", "money", "product"]`.
- Writes the labeled dataset to `artifacts/teacher_dataset.jsonl`, ready for GLiNER fine-tuning.
- When `OPENAI_API_KEY` is missing, it transparently uses GLiNER itself as a pseudo-teacher so the rest of the pipeline still runs (a console notice will tell you which path executed).

### 04 – Benchmark

- Uses a tiny in-repo dataset to compute precision/recall/F1.
- Benchmarks three approaches side by side: local GLiNER inference, GPT teacher-style JSON labeling, and Instructor schema-constrained extraction.
- Includes deployment-footprint numbers for the exported FP32 and INT8 ONNX artifacts to connect model quality with production packaging.
- Writes `artifacts/benchmark_report.md` and `artifacts/benchmark_report.json`, so the repo now has a persistent comparison artifact instead of only transient console logs.

### 05 – Structured extraction

Demonstrates schema-constrained extraction using [Instructor](https://github.com/jxnl/instructor) with a `DocumentNER` Pydantic model. In offline situations the script falls back to GLiNER predictions but still returns the same structured payload.

## 4. Troubleshooting

- **Model downloads blocked?** Ensure you can reach `https://huggingface.co` – GLiNER checkpoints are fetched on first run and cached locally.
- **ONNX export permissions?** The script writes to `artifacts/`; ensure you have write access.
- **OpenAI quota/auth errors?** Double-check the `.env` values and that the key has GPT-4o access. The README instructions above describe where to obtain a key.

Feel free to extend the dataset in the scripts or point them to your own corpora for experimentation. PRs welcome!
