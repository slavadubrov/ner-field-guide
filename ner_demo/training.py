"""Reviewed annotations to GLiNER span-training JSON; no training is claimed."""

import argparse
import json
from pathlib import Path

from ner_demo.core import LABELS, digest, validate_documents
from ner_demo.models import check_gliner_length, load_gliner


def convert_reviewed(records, model, labels=LABELS):
    validate_documents(records, labels)
    splits = {name: [] for name in ("train", "development", "test")}
    text_splits = {}
    for doc in records:
        if (
            doc.get("reviewed") is not True
            or not doc.get("reviewer")
            or doc.get("status") != "ok"
            or doc.get("rejected")
        ):
            raise ValueError(
                "requires a named human review and no unresolved rejection"
            )
        if doc.get("source_sha256") != digest(doc["text"]):
            raise ValueError("source changed since annotation")
        split = doc.get("split")
        if split not in splits:
            raise ValueError("explicit train/development/test split required")
        key = digest(doc["text"])
        if key in text_splits and text_splits[key] != split:
            raise ValueError("same source text occurs across splits")
        text_splits[key] = split
        tokens = list(model.data_processor.words_splitter(doc["text"]))
        if not tokens or len(tokens) > model.config.max_len:
            raise ValueError("empty or truncated training example")
        starts = {start: i for i, (_, start, _) in enumerate(tokens)}
        ends = {end: i for i, (_, _, end) in enumerate(tokens)}
        spans = []
        for entity in doc["entities"]:
            if entity["start"] not in starts or entity["end"] not in ends:
                raise ValueError("entity does not align exactly with training tokens")
            start, end = starts[entity["start"]], ends[entity["end"]]
            if end - start + 1 > model.config.max_width:
                raise ValueError("entity exceeds model span width")
            spans.append(
                [start, end, entity["label"]]
            )  # GLiNER token ends are inclusive.
        splits[split].append(
            dict(
                document_id=doc["document_id"],
                tokenized_text=[t[0] for t in tokens],
                ner=spans,
            )
        )
    return splits


def check_collator(splits, model, labels=LABELS):
    from gliner.data_processing.collator import SpanDataCollator

    collator = SpanDataCollator(
        model.config,
        data_processor=model.data_processor,
        return_tokens=True,
        return_entities=True,
        return_id_to_classes=True,
    )
    checked = 0
    for rows in splits.values():
        for row in rows:
            check_gliner_length(model, " ".join(row["tokenized_text"]), labels)
            batch = collator([row], entity_types=[labels])
            if batch["tokens"][0] != row["tokenized_text"]:
                raise ValueError("collator truncated or changed words")
            if int(batch["labels"].sum()) != len(row["ner"]):
                raise ValueError("collator lost positive training spans")
            for start, end, label in row["ner"]:
                class_id = next(
                    k for k, v in batch["id_to_classes"][0].items() if v == label
                )
                span_index = start * model.config.max_width + end - start
                if batch["labels"][0, span_index, class_id - 1].item() != 1:
                    raise ValueError("collator label alignment mismatch")
            checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/training"))
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in args.annotations.read_text().splitlines()
        if line.strip()
    ]
    model = load_gliner()
    splits = convert_reviewed(records, model)
    if not all(splits.values()):
        raise ValueError("provide non-empty train, development and test splits")
    checked = check_collator(splits, model)
    args.output.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        (args.output / f"{split}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
        )
    print(
        f"Validated {checked} examples through real GLiNER collator. No training was run."
    )


if __name__ == "__main__":
    main()
