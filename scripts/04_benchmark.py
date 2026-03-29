"""Benchmark: GLiNER vs GPT-4o."""
import time
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner-medium-v2.1")
text = "Microsoft acquired LinkedIn in 2016 for $26.2 billion."
labels = ["organization", "money", "date"]
start = time.time()
entities = model.predict_entities(text, labels, threshold=0.5)
elapsed_ms = (time.time() - start) * 1000
print(f"GLiNER: {len(entities)} entities in {elapsed_ms:.0f}ms")
for e in entities:
    print(f"  {e['text']} -> {e['label']}")