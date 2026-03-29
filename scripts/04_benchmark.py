"""Latency + F1 benchmark for GLiNER (local) vs GPT-4o (optional)."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Iterable

from dotenv import load_dotenv
from gliner import GLiNER
from openai import APIError, OpenAI

load_dotenv()

MODEL_ID = "urchade/gliner_medium-v2.1"
LABELS = ["person", "organization", "date", "money", "product", "event"]
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
    return {"text": text, "label": label}


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
    return Metrics(precision, recall, f1)


def benchmark_gliner() -> None:
    print("Loading GLiNER ...")
    model = GLiNER.from_pretrained(MODEL_ID)
    start = time.perf_counter()
    predictions = [
        model.predict_entities(example["text"], LABELS, threshold=0.5)
        for example in DATASET
    ]
    elapsed = (time.perf_counter() - start) * 1000 / len(DATASET)
    metrics = compute_metrics(predictions, [item["entities"] for item in DATASET])
    print(f"GLiNER → latency: {elapsed:.1f} ms/example, F1: {metrics.f1:.2f}")


def benchmark_gpt() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping GPT-4o benchmark (set OPENAI_API_KEY to enable).")
        return

    client = OpenAI()
    preds: list[list[dict]] = []
    latencies: list[float] = []
    for example in DATASET:
        start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Return json with an 'entities' list using only these labels: "
                            + ", ".join(LABELS)
                        ),
                    },
                    {"role": "user", "content": example["text"]},
                ],
            )
            payload = json.loads(response.choices[0].message.content)
            cleaned = [
                coerced
                for ent in payload.get("entities", [])
                if (coerced := coerce_entity(ent))
            ]
            preds.append(cleaned)
        except APIError as exc:
            print(f"OpenAI error: {exc}")
            preds.append([])
        finally:
            latencies.append((time.perf_counter() - start) * 1000)

    metrics = compute_metrics(preds, [item["entities"] for item in DATASET])
    avg_latency = sum(latencies) / len(latencies)
    print(f"GPT-4o → latency: {avg_latency:.1f} ms/example, F1: {metrics.f1:.2f}")


if __name__ == "__main__":
    benchmark_gliner()
    benchmark_gpt()
