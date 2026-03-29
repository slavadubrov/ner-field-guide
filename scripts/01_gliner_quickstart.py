"""GLiNER zero-shot NER in 10 lines."""
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
text = "Bill Gates founded Microsoft on April 4, 1975."
labels = ["person", "organization", "date"]
entities = model.predict_entities(text, labels, threshold=0.5)
for e in entities:
    print(f"{e['text']:30} -> {e['label']}")