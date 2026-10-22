"""Cached pinned inference on a few sentences; opt in before downloading weights."""

import argparse

from ner_demo.core import LABELS, example_documents
from ner_demo.models import make_runner
from ner_demo.registry import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="gliner",
        choices=[
            name
            for name, entry in MODELS.items()
            if entry["task"] == "NER" and entry["adapter"] != "openai"
        ],
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--examples",
        action="store_true",
        help="run the English, German, and Russian functional examples",
    )
    args = parser.parse_args()
    predict = make_runner(args.model, LABELS, args.download)
    if not args.examples:
        print(predict("Bill Gates founded Microsoft."))
        return
    for doc in example_documents():
        found = predict(doc["text"])
        gold = {(e["start"], e["end"], e["label"]) for e in doc["entities"]}
        hits = len(gold & {(e["start"], e["end"], e["label"]) for e in found})
        print(f"{doc['document_id']}: {hits}/{len(gold)} gold spans found")
        for e in sorted(found, key=lambda e: (e["start"], e["end"])):
            score = f" {e['score']:.2f}" if "score" in e else ""
            print(f"  [{e['start']}:{e['end']}] {e['text']!r} {e['label']}{score}")
    print("Functional examples only; three sentences are not an accuracy benchmark.")


if __name__ == "__main__":
    main()
