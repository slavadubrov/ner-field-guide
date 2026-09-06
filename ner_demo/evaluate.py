"""One runner for smoke and versioned local dataset bundles."""

import argparse
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import time
from pathlib import Path

from ner_demo.core import LABELS, digest, score, smoke_documents, validate_documents
from ner_demo.models import BACKBONES, PINS, RULES, make_runner


def evaluate(documents, labels, factory, repeats=3, warmups=1):
    if repeats < 1 or warmups < 0:
        raise ValueError("repeats must be positive; warmups nonnegative")
    validate_documents(documents, labels)
    started = time.perf_counter()
    try:
        runner = factory()
        init_error = None
    except Exception as exc:
        runner, init_error = None, type(exc).__name__
    initialization_ms = 1000 * (time.perf_counter() - started)
    warmup_errors = []
    if runner:
        for _ in range(warmups):
            try:
                runner(documents[0]["text"])
            except Exception as exc:
                warmup_errors.append(type(exc).__name__)
    runs = []
    for repeat in range(repeats):
        records = []
        for doc in documents:
            row = dict(
                document_id=doc["document_id"],
                status="error",
                entities=[],
                latency_ms=None,
            )
            if init_error:
                row["error"] = "initialization:" + init_error
            else:
                start = time.perf_counter()
                try:
                    prediction = runner(doc["text"])
                    if isinstance(prediction, dict):
                        row["provenance"] = prediction
                        row["entities"] = prediction["entities"]
                        row["status"] = (
                            "ok" if prediction["status"] == "ok" else "error"
                        )
                    else:
                        row["entities"] = prediction
                        row["status"] = "ok"
                except Exception as exc:
                    # Provider errors may contain request text or credentials.
                    row["error"] = type(exc).__name__
                row["latency_ms"] = 1000 * (time.perf_counter() - start)
            records.append(row)
        runs.append(
            dict(
                repeat=repeat,
                records=records,
                metrics=score(documents, records, labels),
            )
        )
    timings = sorted(
        r["latency_ms"] for run in runs for r in run["records"] if r["status"] == "ok"
    )
    return dict(
        initialization_ms=initialization_ms,
        warmup_errors=warmup_errors,
        runs=runs,
        latency_ms=dict(
            p50=statistics.median(timings) if timings else None,
            p95=timings[max(0, math.ceil(0.95 * len(timings)) - 1)]
            if timings
            else None,
            successful_calls=len(timings),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["rules", "encoder", "gliner", "gliner25", "llm"],
        default=["rules"],
    )
    parser.add_argument(
        "--download", action="store_true", help="Explicitly allow model downloads"
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/evaluation.json")
    )
    args = parser.parse_args()
    if args.models != list(dict.fromkeys(args.models)) or not 0 <= args.threshold <= 1:
        parser.error("models must be unique and threshold must be in [0,1]")
    if args.dataset:
        bundle = json.loads(args.dataset.read_text())
        documents, labels = bundle["documents"], bundle["manifest"]["labels"]
        if bundle["manifest"]["documents_sha256"] != digest(documents):
            raise ValueError("dataset content hash mismatch")
    else:
        documents, labels = smoke_documents(), LABELS
        bundle = {
            "manifest": {"kind": "synthetic regression smoke; not a model ranking"}
        }
    if "llm" in args.models:
        if (
            not args.allow_paid
            or args.repeats != 1
            or args.warmups != 0
            or len(documents) > 32
        ):
            parser.error(
                "LLM requires --allow-paid --repeats 1 --warmups 0 and at most 32 documents"
            )
        if any(len(d["text"]) > 4000 for d in documents):
            parser.error("LLM demo caps each document at 4000 characters")
    manifest = dict(
        dataset=bundle["manifest"],
        document_ids=[d["document_id"] for d in documents],
        document_hashes={d["document_id"]: digest(d) for d in documents},
        character_lengths=[len(d["text"]) for d in documents],
        labels=labels,
        batch_size=1,
        label_count=len(labels),
        device=args.device,
        repeats=args.repeats,
        warmups=args.warmups,
        threshold=args.threshold,
        flat_ner=True,
        truncation="reject, retain failed documents in denominator",
        platform=platform.platform(),
        processor=platform.processor(),
        python=platform.python_version(),
        rules=RULES,
        gliner_backbones=BACKBONES,
        models={
            name: PINS.get(
                name,
                ("openai:gpt-5.4-mini", "returned model recorded per call")
                if name == "llm"
                else ("literal rules", digest(RULES)),
            )
            for name in args.models
        },
    )
    manifest["packages"] = {}
    for package in [
        "ner-field-guide",
        "gliner",
        "torch",
        "transformers",
        "onnxruntime",
        "openai",
    ]:
        try:
            manifest["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    manifest["source_sha256"] = digest(
        {p.name: p.read_text() for p in Path(__file__).parent.glob("*.py")}
    )
    manifest["git_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    results = {
        name: evaluate(
            documents,
            labels,
            lambda name=name: make_runner(
                name, labels, args.download, args.device, args.threshold
            ),
            args.repeats,
            args.warmups,
        )
        for name in args.models
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(manifest=manifest, results=results), indent=2, ensure_ascii=False
        )
        + "\n"
    )
    for name, result in results.items():
        metrics = result["runs"][0]["metrics"]
        print(
            name,
            json.dumps(
                dict(
                    micro=metrics["micro"],
                    completion=metrics["completion"],
                    headline=metrics["headline"],
                )
            ),
        )
    print(args.output)


if __name__ == "__main__":
    main()
