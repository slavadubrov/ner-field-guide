"""Structured extraction with Instructor and a configurable OpenAI model."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from gliner import GLiNER
from instructor import from_openai
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

MODEL_ID = "urchade/gliner_medium-v2.1"
LABELS = ["person", "organization", "date", "location"]
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
_fallback_model: GLiNER | None = None


class EntityMention(BaseModel):
    text: str = Field(description="Entity text")
    label: str = Field(description="Entity type")


class DocumentNER(BaseModel):
    entities: list[EntityMention]


def run_with_instructor(text: str) -> DocumentNER:
    if not OPENAI_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for Instructor client.")

    client = from_openai(OpenAI(api_key=OPENAI_KEY))
    return client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": f"Extract entities from: {text}"}],
        response_model=DocumentNER,
    )


def fallback_with_gliner(text: str) -> DocumentNER:
    global _fallback_model
    if _fallback_model is None:
        print("OPENAI_API_KEY not set → using GLiNER fallback.")
        _fallback_model = GLiNER.from_pretrained(MODEL_ID)
    model = _fallback_model
    preds = model.predict_entities(text, LABELS, threshold=0.5)
    return DocumentNER(
        entities=[EntityMention(text=p["text"], label=p["label"]) for p in preds]
    )


def main() -> None:
    text = "Apple Inc. was founded in Cupertino by Steve Jobs in 1976."
    if not OPENAI_KEY:
        result = fallback_with_gliner(text)
    else:
        result = run_with_instructor(text)
        print(f"Using {OPENAI_MODEL} for structured extraction.")
    print(result.model_dump())


if __name__ == "__main__":
    main()
