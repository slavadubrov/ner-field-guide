"""Exact occurrence scoring. Offsets are Python Unicode indices, [start, end)."""

import hashlib
import json
from collections import Counter

LABELS = ["person", "organization", "location"]


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def validate_span(text, entity, labels):
    if not isinstance(entity, dict):
        raise ValueError("span must be an object")
    start, end = entity.get("start"), entity.get("end")
    if (
        type(start) is not int
        or type(end) is not int
        or not 0 <= start < end <= len(text)
    ):
        raise ValueError("invalid offsets")
    if not isinstance(entity.get("label"), str) or entity["label"] not in labels:
        raise ValueError("unknown label")
    if entity.get("text") != text[start:end]:
        raise ValueError("span text does not match source")
    return start, end, entity["label"]


def validate_documents(documents, labels):
    if (
        not labels
        or any(not isinstance(label, str) or not label for label in labels)
        or len(labels) != len(set(labels))
    ):
        raise ValueError("labels must be unique nonempty strings")
    by_id = {}
    for doc in documents:
        doc_id = doc["document_id"]
        if not isinstance(doc_id, str) or not doc_id or doc_id in by_id:
            raise ValueError("missing or duplicate document ID")
        if not isinstance(doc["text"], str) or not isinstance(doc["entities"], list):
            raise ValueError("invalid document")
        keys = [validate_span(doc["text"], e, labels) for e in doc["entities"]]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate gold occurrence")
        by_id[doc_id] = doc
    if not by_id:
        raise ValueError("empty evaluation dataset")
    return by_id


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        support=tp + fn,
        precision=p,
        recall=r,
        f1=2 * p * r / (p + r) if p + r else 0.0,
    )


def score(documents, records, labels):
    """Missing/failed/skipped documents remain FN; duplicate spans add FP.

    Unknown/duplicate result document IDs are protocol errors, never silently
    scored. Invalid predictions add FP and cannot match gold. Headline metrics
    are withheld if any planned document did not produce a valid complete result.
    """
    gold = validate_documents(documents, labels)
    results = {}
    for row in records:
        key = row["document_id"]
        if key not in gold or key in results:
            raise ValueError("extra or duplicate prediction document")
        if row["status"] not in {"ok", "error", "skipped"}:
            raise ValueError("unknown result status")
        if not isinstance(row["entities"], list):
            raise ValueError("entities must be a list")
        results[key] = row
    counts = {label: [0, 0, 0] for label in labels}
    invalid = completed = valid_docs = 0
    for doc_id, doc in gold.items():
        target = Counter(validate_span(doc["text"], e, labels) for e in doc["entities"])
        row = results.get(doc_id, {"status": "skipped", "entities": []})
        completed += row["status"] == "ok"
        predicted = Counter()
        bad = 0
        # Preserve partial predictions on errors too: they can add FP, but not TP.
        for entity in row["entities"]:
            try:
                key = validate_span(doc["text"], entity, labels)
                if row["status"] == "ok":
                    predicted[key] += 1
                else:
                    counts[key[2]][1] += 1
            except ValueError:
                bad += 1
                label = entity.get("label") if isinstance(entity, dict) else None
                if isinstance(label, str) and label in counts:
                    counts[label][1] += 1
                else:
                    invalid += 1
        valid_docs += row["status"] == "ok" and bad == 0
        for key in target.keys() | predicted.keys():
            tp = min(target[key], predicted[key])
            c = counts[key[2]]
            c[0] += tp
            c[1] += predicted[key] - tp
            c[2] += target[key] - tp
    per_label = {label: prf(*c) for label, c in counts.items()}
    total = [sum(c[i] for c in counts.values()) for i in range(3)]
    total[1] += invalid
    micro = prf(*total)
    macro = {
        k: sum(m[k] for m in per_label.values()) / len(labels)
        for k in ("precision", "recall", "f1")
    }
    return dict(
        micro=micro,
        macro=macro,
        per_label=per_label,
        headline=micro if valid_docs == len(gold) else None,
        planned_documents=len(gold),
        completed_documents=completed,
        valid_documents=valid_docs,
        completion=completed / len(gold),
        scoring_coverage=valid_docs / len(gold),
        unknown_label_fp=invalid,
    )


def smoke_documents():
    return [
        {
            "document_id": "smoke:repeat",
            "text": "John met John.",
            "entities": [
                {"start": 0, "end": 4, "text": "John", "label": "person"},
                {"start": 9, "end": 13, "text": "John", "label": "person"},
            ],
        },
        {
            "document_id": "smoke:unicode",
            "text": "Zoë works at Microsoft in Paris.",
            "entities": [
                {"start": 0, "end": 3, "text": "Zoë", "label": "person"},
                {"start": 13, "end": 22, "text": "Microsoft", "label": "organization"},
                {"start": 26, "end": 31, "text": "Paris", "label": "location"},
            ],
        },
        {"document_id": "smoke:empty", "text": "Nothing to extract!", "entities": []},
    ]
