"""LLM-as-teacher pipeline: label text with GPT-4o (or GLiNER fallback)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from gliner import GLiNER
from openai import APIError, OpenAI

load_dotenv()

MODEL_ID = "urchade/gliner_medium-v2.1"
ENTITY_TYPES = ["person", "organization", "date", "money", "product"]
UNLABELED_SNIPPETS = [
    "NVIDIA acquired Arm for $40 billion in 2020.",
    "Anthropic introduced Claude 3 in March 2024.",
    "Amazon invested $4B in Anthropic in 2023.",
]
ARTIFACTS = Path("artifacts")
OUTPUT_PATH = ARTIFACTS / "teacher_dataset.jsonl"

client = OpenAI() if os.getenv("OPENAI_API_KEY") else None
_fallback_model: GLiNER | None = None


def _fallback_predict(text: str) -> list[dict]:
    global _fallback_model
    if _fallback_model is None:
        print("→ Using GLiNER as a local pseudo-teacher (OPENAI_API_KEY missing).")
        _fallback_model = GLiNER.from_pretrained(MODEL_ID)
    preds = _fallback_model.predict_entities(text, ENTITY_TYPES, threshold=0.45)
    return [{"text": p["text"], "type": p["label"]} for p in preds]


def label_with_llm(text: str) -> list[dict]:
    if client is None:
        return _fallback_predict(text)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": f"Extract entities. Allowed types: {', '.join(ENTITY_TYPES)}.",
                },
                {"role": "user", "content": text},
            ],
        )
    except APIError as exc:
        print(f"OpenAI error ({exc}); falling back to GLiNER.")
        return _fallback_predict(text)

    payload = json.loads(response.choices[0].message.content)
    return payload.get("entities", [])


def build_dataset(snippets: Iterable[str]) -> list[dict]:
    dataset = []
    for snippet in snippets:
        dataset.append({"text": snippet, "entities": label_with_llm(snippet)})
    return dataset


def save_dataset(examples: Iterable[dict]) -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    with OUTPUT_PATH.open("w") as fp:
        for example in examples:
            fp.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    dataset = build_dataset(UNLABELED_SNIPPETS)
    save_dataset(dataset)
    print(f"Wrote {len(dataset)} labeled examples to {OUTPUT_PATH}")
