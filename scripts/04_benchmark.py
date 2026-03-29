"""Generate a side-by-side benchmark report for the main NER approaches in the repo."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from dotenv import load_dotenv
from gliner import GLiNER
from instructor import from_openai
from openai import APIError, OpenAI
from pydantic import BaseModel, Field

load_dotenv()

MODEL_ID = "urchade/gliner_medium-v2.1"
ARTIFACTS_DIR = Path("artifacts")
FP32_MODEL_PATH = ARTIFACTS_DIR / "gliner_medium.onnx" / "model.onnx"
INT8_MODEL_PATH = ARTIFACTS_DIR / "gliner_medium_int8.onnx"
REPORT_JSON = ARTIFACTS_DIR / "benchmark_report.json"
REPORT_MD = ARTIFACTS_DIR / "benchmark_report.md"
LABELS = ["person", "organization", "date", "money", "product", "event"]
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

DATASET = [
    {
        "text": "Microsoft acquired LinkedIn in 2016 for $26.2 billion.",
        "entities": [
            {"text": "Microsoft", "label": "organization"},
            {"text": "LinkedIn", "label": "organization"},
            {"text": "2016", "label": "date"},
            {"text": "$26.2 billion", "label": "money"},
        ],
    },
    {
        "text": "Apple announced the Vision Pro at WWDC 2023.",
        "entities": [
            {"text": "Apple", "label": "organization"},
            {"text": "Vision Pro", "label": "product"},
            {"text": "WWDC 2023", "label": "event"},
        ],
    },
    {
        "text": "OpenAI raised $10B from Microsoft in 2023.",
        "entities": [
            {"text": "OpenAI", "label": "organization"},
            {"text": "$10B", "label": "money"},
            {"text": "Microsoft", "label": "organization"},
            {"text": "2023", "label": "date"},
        ],
    },
]


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float


@dataclass
class BenchmarkResult:
    name: str
    average_latency_ms: float | None
    metrics: Metrics | None
    notes: str


class EntityMention(BaseModel):
    text: str = Field(description="Entity text")
    label: str = Field(description="Entity type")


class DocumentNER(BaseModel):
    entities: list[EntityMention]


def coerce_entity(entity: dict) -> dict | None:
    text = (
        entity.get("text")
        or entity.get("entity")
        or entity.get("span")
        or entity.get("mention")
    )
    label = (
        entity.get("label")
        or entity.get("type")
        or entity.get("category")
        or entity.get("class")
    )
    if not text or not label:
        return None
    return {"text": str(text), "label": str(label)}


def normalize(entity: dict) -> tuple[str, str]:
    return entity["text"].strip().lower(), entity["label"].strip().lower()


def compute_metrics(
    predictions: Iterable[list[dict]], gold: Iterable[list[dict]]
) -> Metrics:
    tp = fp = fn = 0
    for pred, target in zip(predictions, gold):
        pred_set = {normalize(e) for e in pred}
        gold_set = {normalize(e) for e in target}
        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )
    return Metrics(precision=precision, recall=recall, f1=f1)


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def measure_runner(
    name: str, runner: Callable[[str], list[dict]], notes: str
) -> BenchmarkResult:
    predictions: list[list[dict]] = []
    latencies_ms: list[float] = []
    for example in DATASET:
        start = time.perf_counter()
        predictions.append(runner(example["text"]))
        latencies_ms.append((time.perf_counter() - start) * 1000)
    metrics = compute_metrics(predictions, [item["entities"] for item in DATASET])
    return BenchmarkResult(
        name=name,
        average_latency_ms=average(latencies_ms),
        metrics=metrics,
        notes=notes,
    )


def benchmark_gliner() -> BenchmarkResult:
    print("Loading GLiNER ...")
    model = GLiNER.from_pretrained(MODEL_ID)

    def runner(text: str) -> list[dict]:
        return model.predict_entities(text, LABELS, threshold=0.5)

    return measure_runner(
        name="GLiNER (PyTorch)",
        runner=runner,
        notes="Best production inference path in this repo: local, fast, and deterministic.",
    )


def benchmark_gpt_teacher(client: OpenAI) -> BenchmarkResult:
    def runner(text: str) -> list[dict]:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract entities and return json with "
                        '{"entities": [{"text": "...", "type": "..."}]}. '
                        f"Allowed types: {', '.join(LABELS)}."
                    ),
                },
                {"role": "user", "content": text},
            ],
        )
        payload = json.loads(response.choices[0].message.content)
        return [
            coerced
            for ent in payload.get("entities", [])
            if (coerced := coerce_entity(ent))
        ]

    return measure_runner(
        name=f"{OPENAI_MODEL} teacher labeling",
        runner=runner,
        notes="High-quality labels for dataset generation, but network latency dominates.",
    )


def benchmark_instructor(client: OpenAI) -> BenchmarkResult:
    instructor_client = from_openai(client)

    def runner(text: str) -> list[dict]:
        result = instructor_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Extract named entities from this text using the schema. "
                        f"Allowed labels: {', '.join(LABELS)}.\n\n{text}"
                    ),
                }
            ],
            response_model=DocumentNER,
        )
        return [{"text": item.text, "label": item.label} for item in result.entities]

    return measure_runner(
        name=f"{OPENAI_MODEL} structured extraction",
        runner=runner,
        notes="Most ergonomic app integration path, but slower than encoder inference.",
    )


def filesize_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    return path.stat().st_size / (1024 * 1024)


def build_footprint_section() -> list[dict]:
    fp32_mb = filesize_mb(FP32_MODEL_PATH)
    int8_mb = filesize_mb(INT8_MODEL_PATH)
    reduction_pct = None
    if fp32_mb and int8_mb:
        reduction_pct = (1 - (int8_mb / fp32_mb)) * 100
    return [
        {
            "artifact": "GLiNER ONNX FP32",
            "path": str(FP32_MODEL_PATH),
            "size_mb": round(fp32_mb, 1) if fp32_mb is not None else None,
            "notes": "Export generated by scripts/02_onnx_export.py",
        },
        {
            "artifact": "GLiNER ONNX INT8",
            "path": str(INT8_MODEL_PATH),
            "size_mb": round(int8_mb, 1) if int8_mb is not None else None,
            "notes": (
                f"{reduction_pct:.1f}% smaller than FP32"
                if reduction_pct is not None
                else "Run scripts/02_onnx_export.py first to populate this artifact"
            ),
        },
    ]


def render_table(results: list[BenchmarkResult]) -> str:
    rows = [
        [
            result.name,
            f"{result.average_latency_ms:.1f}"
            if result.average_latency_ms is not None
            else "n/a",
            f"{result.metrics.precision:.2f}" if result.metrics else "n/a",
            f"{result.metrics.recall:.2f}" if result.metrics else "n/a",
            f"{result.metrics.f1:.2f}" if result.metrics else "n/a",
            result.notes,
        ]
        for result in results
    ]
    headers = ["Approach", "Avg latency (ms)", "Precision", "Recall", "F1", "Notes"]
    widths = [
        max(len(str(cell)) for cell in [header] + [row[idx] for row in rows])
        for idx, header in enumerate(headers)
    ]
    header_line = " | ".join(
        header.ljust(widths[idx]) for idx, header in enumerate(headers)
    )
    separator = "-+-".join("-" * width for width in widths)
    body = [
        " | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row))
        for row in rows
    ]
    return "\n".join([header_line, separator] + body)


def write_report(results: list[BenchmarkResult], footprint: list[dict]) -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    payload = {
        "model": OPENAI_MODEL,
        "dataset_size": len(DATASET),
        "results": [
            {
                "name": result.name,
                "average_latency_ms": round(result.average_latency_ms or 0.0, 1),
                "precision": round(result.metrics.precision, 3)
                if result.metrics
                else None,
                "recall": round(result.metrics.recall, 3) if result.metrics else None,
                "f1": round(result.metrics.f1, 3) if result.metrics else None,
                "notes": result.notes,
            }
            for result in results
        ],
        "footprint": footprint,
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    markdown_lines = [
        "# Benchmark Report",
        "",
        f"- Dataset size: {len(DATASET)} examples",
        f"- OpenAI model: `{OPENAI_MODEL}`",
        "",
        "## Inference Comparison",
        "",
        "| Approach | Avg latency (ms) | Precision | Recall | F1 | Notes |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        markdown_lines.append(
            "| "
            + " | ".join(
                [
                    result.name,
                    f"{result.average_latency_ms:.1f}"
                    if result.average_latency_ms is not None
                    else "n/a",
                    f"{result.metrics.precision:.2f}" if result.metrics else "n/a",
                    f"{result.metrics.recall:.2f}" if result.metrics else "n/a",
                    f"{result.metrics.f1:.2f}" if result.metrics else "n/a",
                    result.notes,
                ]
            )
            + " |"
        )
    markdown_lines.extend(
        [
            "",
            "## Deployment Footprint",
            "",
            "| Artifact | Size (MB) | Notes |",
            "| --- | ---: | --- |",
        ]
    )
    for entry in footprint:
        size = f"{entry['size_mb']:.1f}" if entry["size_mb"] is not None else "n/a"
        markdown_lines.append(f"| {entry['artifact']} | {size} | {entry['notes']} |")
    REPORT_MD.write_text("\n".join(markdown_lines) + "\n")


def main() -> None:
    results = [benchmark_gliner()]

    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
        try:
            results.append(benchmark_gpt_teacher(client))
            results.append(benchmark_instructor(client))
        except APIError as exc:
            print(f"OpenAI benchmark failed: {exc}")
    else:
        print("Skipping OpenAI comparisons because OPENAI_API_KEY is not set.")

    footprint = build_footprint_section()

    print()
    print(render_table(results))
    print()
    for entry in footprint:
        size = f"{entry['size_mb']:.1f} MB" if entry["size_mb"] is not None else "n/a"
        print(f"{entry['artifact']}: {size} ({entry['notes']})")

    write_report(results, footprint)
    print()
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
