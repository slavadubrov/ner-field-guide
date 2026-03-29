# ner-field-guide

Companion code for "The Definitive Guide to NER in 2026".

Run:

    uv sync
    uv run python scripts/01_gliner_quickstart.py

## Scripts

| Script | Description |
|--------|-------------|
| `01_gliner_quickstart.py` | GLiNER zero-shot in 10 lines |
| `02_onnx_export.py` | ONNX export + INT8 |
| `03_llm_teacher_pipeline.py` | LLM-as-teacher labeling |
| `04_benchmark.py` | F1 + latency comparison |
| `05_structured_extraction.py` | Instructor structured extraction |

## Requirements

- Python 3.11+
- uv (pip install uv)
