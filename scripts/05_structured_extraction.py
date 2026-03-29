"""Structured extraction with Instructor."""
from instructor import from_openai
from pydantic import BaseModel, Field
from openai import OpenAI

client = from_openai(OpenAI())

class EntityMention(BaseModel):
    text: str = Field(description="Entity text")
    label: str = Field(description="Entity type")

class DocumentNER(BaseModel):
    entities: list[EntityMention]

text = "Apple Inc. was founded in Cupertino by Steve Jobs in 1976."
result = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": f"Extract entities from: {text}"}],
    response_model=DocumentNER
)
print(result.model_dump())