"""LLM-as-teacher: label with GPT-4o, fine-tune GLiNER."""
from openai import OpenAI
import json

client = OpenAI()
ENTITY_TYPES = ["person", "organization", "date", "money", "product"]

def label_with_llm(text: str) -> list[dict]:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "system",
            "content": f"""Extract entities. Return JSON: {{"entities": [{{"text": "...", "type": "..."}}]}}
Types: {", ".join(ENTITY_TYPES)}"""
        }, {"role": "user", "content": text}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# Example usage
text = "NVIDIA acquired Arm for $40 billion in 2020."
result = label_with_llm(text)
print(result)