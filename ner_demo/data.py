"""Import a bounded, pinned Few-NERD supervised slice without dataset scripts."""

import argparse
import hashlib
import io
import json
import urllib.request
from pathlib import Path

from ner_demo.core import LABELS, digest, load_bundle, validate_documents

REPO = "DFKI-SLT/few-nerd"
REVISION = "205f3e9c9f3577ea2561d43f2f62dc249ab92d5b"
# Coarse IDs from this revision's dataset metadata. All other categories excluded
# from the explicit PER/ORG/LOC projection, never relabeled as one of these types.
MAPPING = {4: "location", 5: "organization", 7: "person"}


def convert_row(row, split, index):
    tokens, tags, fine = row["tokens"], row["ner_tags"], row["fine_ner_tags"]
    if len(tokens) != len(tags) or len(tokens) != len(fine) or not tokens:
        raise ValueError("invalid token/tag cardinality")
    text = " ".join(tokens)
    starts, cursor = [], 0
    for token in tokens:
        starts.append(cursor)
        cursor += len(token) + 1
    entities = []
    i = 0
    while i < len(tokens):
        j = i + 1
        while j < len(tokens) and fine[j] == fine[i]:
            j += 1
        if tags[i] in MAPPING:
            if any(tags[k] != tags[i] for k in range(i, j)):
                raise ValueError("inconsistent coarse/fine tags")
            start, end = starts[i], starts[j - 1] + len(tokens[j - 1])
            entities.append(
                dict(start=start, end=end, text=text[start:end], label=MAPPING[tags[i]])
            )
        i = j
    return dict(
        document_id=f"few-nerd:supervised:{split}:{row.get('id', index)}",
        text=text,
        entities=entities,
    )


def select(rows, split, limit, exclude=frozenset()):
    """First `limit` converted rows whose text is not in `exclude`."""
    docs, skipped = [], 0
    for index, row in enumerate(rows):
        if len(docs) == limit:
            break
        doc = convert_row(row, split, index)
        if doc["text"] in exclude:
            skipped += 1
        else:
            docs.append(doc)
    return docs, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=["train", "validation", "test"], default="test"
    )
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument(
        "--output", type=Path, help="default: artifacts/few-nerd-<split>.json"
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        nargs="+",
        default=[],
        metavar="BUNDLE",
        help="skip rows whose text occurs in these imported files; Few-NERD "
        "repeats some sentences across splits",
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    # One file per split, so a validation import cannot overwrite the test file.
    args.output = args.output or Path(f"artifacts/few-nerd-{args.split}.json")
    import pyarrow.parquet as pq

    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/supervised/{args.split}-00000-of-00001.parquet"
    with urllib.request.urlopen(url, timeout=60) as response:
        raw = response.read(100_000_001)
    if len(raw) > 100_000_000:
        raise ValueError("dataset file exceeds 100 MB download cap")
    table = pq.read_table(io.BytesIO(raw))
    bundles = [load_bundle(path) for path in args.exclude]
    exclude = {d["text"] for bundle in bundles for d in bundle["documents"]}
    # Lazy, so selection stops after `limit` kept rows however many are skipped.
    batches = table.to_batches(max_chunksize=1000)
    rows = (row for batch in batches for row in batch.to_pylist())
    docs, skipped = select(rows, args.split, args.limit, exclude)
    validate_documents(docs, LABELS)
    manifest = dict(
        dataset=REPO,
        revision=REVISION,
        split=args.split,
        config="supervised",
        license="CC-BY-SA-4.0",
        source_url=url,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        labels=LABELS,
        label_mapping=MAPPING,
        selection="first N rows in pinned split"
        + (", skipping rows whose text is in `excluded`" if exclude else ""),
        limit=args.limit,
        excluded=[
            dict(
                file=str(path), documents_sha256=bundle["manifest"]["documents_sha256"]
            )
            for path, bundle in zip(args.exclude, bundles)
        ],
        excluded_rows=skipped,
        split_rows=table.num_rows,
        documents=len(docs),
        text_construction="tokens joined by one ASCII space; not original Wikipedia offsets",
        documents_sha256=digest(docs),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(manifest=manifest, documents=docs), ensure_ascii=False, indent=2
        )
        + "\n"
    )
    print(f"Imported {len(docs)} documents into {args.output}")


if __name__ == "__main__":
    main()
