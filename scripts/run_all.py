"""Convenience CLI to run every demo script sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

SCRIPT_SEQUENCE = [
    ("01_gliner_quickstart.py", "GLiNER quickstart"),
    ("02_onnx_export.py", "ONNX export + INT8"),
    ("03_llm_teacher_pipeline.py", "LLM teacher pipeline"),
    ("04_benchmark.py", "Latency + F1 benchmark"),
    ("05_structured_extraction.py", "Structured extraction"),
]


def run_script(script_name: str) -> int:
    script_path = SCRIPTS_DIR / script_name
    print(f"\n==> Running {script_name}")
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    if result.returncode == 0:
        print(f"✓ {script_name} completed")
    else:
        print(f"✗ {script_name} failed with exit code {result.returncode}")
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ignore-errors",
        action="store_true",
        help="Run every script even if an earlier one fails (default: stop on first failure).",
    )
    args = parser.parse_args()

    for script_name, _ in SCRIPT_SEQUENCE:
        code = run_script(script_name)
        if code != 0 and not args.ignore_errors:
            sys.exit(code)


if __name__ == "__main__":
    main()
