"""Score adapters on the same pinned documents, labels, and scoring rules."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

from ner_demo.core import (
    LABELS,
    digest,
    f1_interval,
    load_bundle,
    score,
    smoke_documents,
    validate_documents,
)
from ner_demo.models import RULES, make_runner
from ner_demo.registry import MODELS
from ner_demo.report import table

# Adapters whose output depends on a span-score threshold.
THRESHOLDED = {"gliner", "gliner-bi", "gliner2"}
PACKAGES = [
    "ner-field-guide",
    "gliner",
    "gliner2",
    "torch",
    "transformers",
    "huggingface-hub",
    "onnx",
    "onnxruntime",
    "openai",
]


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
    # Warmup calls are the cold start: timed, but kept out of p50/p95.
    warmup_ms, warmup_errors = [], []
    if runner:
        for _ in range(warmups):
            start = time.perf_counter()
            try:
                runner(documents[0]["text"])
            except Exception as exc:
                warmup_errors.append(type(exc).__name__)
            warmup_ms.append(1000 * (time.perf_counter() - start))
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
                # The adapter never ran on this document: missing, not failed.
                row.update(status="skipped", error="initialization:" + init_error)
            else:
                start = time.perf_counter()
                try:
                    prediction = runner(doc["text"])
                    if isinstance(prediction, dict):
                        # An LLM record that finished but had bad spans is scored
                        # like any adapter output: the rejected spans reach the
                        # scorer as false positives. Refusals, truncated output,
                        # and API errors (no span attached) fail the document.
                        row["provenance"] = prediction
                        spans = [
                            r["entity"] for r in prediction["rejected"] if "entity" in r
                        ]
                        finished = prediction["status"] == "ok" or (
                            prediction["status"] == "invalid"
                            and len(spans) == len(prediction["rejected"])
                        )
                        row["entities"] = prediction["entities"] + spans
                        row["status"] = "ok" if finished else "error"
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
        initialization_error=init_error,
        label_cache_ms=getattr(runner, "label_cache_ms", None),
        warmup_ms=warmup_ms,
        warmup_errors=warmup_errors,
        runs=runs,
        # Summaries use the first run; flag repeats that disagree with it.
        metrics_vary_across_repeats=len(
            {json.dumps(run["metrics"]["micro"], sort_keys=True) for run in runs}
        )
        > 1,
        latency_ms=dict(
            p50=statistics.median(timings) if timings else None,
            p95=timings[max(0, math.ceil(0.95 * len(timings)) - 1)]
            if timings
            else None,
            successful_calls=len(timings),
        ),
    )


def cpu_model():
    try:
        if sys.platform == "darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            ).stdout.strip()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None


def git_state():
    """Commit of this source tree; None when it is not a git checkout.

    An installed copy often sits inside another repository (a project's
    .venv), and that repository's commit says nothing about this code.
    """

    def git(*command):
        return subprocess.run(
            ["git", *command],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )

    unknown = dict(git_commit=None, git_dirty=None)
    try:
        if git("ls-files", "--error-unmatch", Path(__file__).name).returncode:
            return unknown
        commit = git("rev-parse", "HEAD").stdout.strip()
        status = git("status", "--porcelain", "--untracked-files=no").stdout
    except OSError:  # git is not installed
        return unknown
    return dict(git_commit=commit, git_dirty=bool(status.strip()))


def environment():
    env = dict(
        platform=platform.platform(),
        machine=platform.machine(),
        cpu=cpu_model(),
        logical_cpus=os.cpu_count(),
        python=platform.python_version(),
    )
    torch = sys.modules.get("torch")
    if torch:
        env["torch_threads"] = torch.get_num_threads()
    env["packages"] = {}
    for package in PACKAGES:
        try:
            env["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    env["source_sha256"] = digest(
        {p.name: p.read_text() for p in Path(__file__).parent.glob("*.py")}
    )
    return env | git_state()


def model_config(name, threshold):
    entry = MODELS[name]
    config = {
        key: entry[key]
        for key in ("family", "role", "checkpoint", "revision", "adapter", "labels")
    }
    for key in ("backbone", "labels_tokenizer", "label_cache"):
        if key in entry:
            config[key] = entry[key]
    if name == "rules":
        config["rules"] = RULES
    config["threshold"] = threshold if entry["adapter"] in THRESHOLDED else None
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[name for name, entry in MODELS.items() if entry["task"] == "NER"],
        default=["rules"],
    )
    parser.add_argument(
        "--download", action="store_true", help="Explicitly allow model downloads"
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="span-score threshold for GLiNER-family adapters",
    )
    parser.add_argument(
        "--sweep",
        type=float,
        nargs="+",
        metavar="T",
        help="development data only: score each threshold and select the best "
        "micro F1 per model (ties go to the first listed)",
    )
    parser.add_argument(
        "--thresholds-from",
        type=Path,
        metavar="SWEEP_JSON",
        help="per-model thresholds selected by a --sweep run on other documents",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/evaluation.json")
    )
    args = parser.parse_args()
    if args.models != list(dict.fromkeys(args.models)):
        parser.error("models must be unique")
    if not all(0 <= t <= 1 for t in [args.threshold, *(args.sweep or [])]):
        parser.error("thresholds must be in [0,1]")
    if args.sweep and args.thresholds_from:
        parser.error("use --sweep on development data or --thresholds-from, not both")
    if args.dataset:
        bundle = load_bundle(args.dataset)
        documents, labels = bundle["documents"], bundle["manifest"]["labels"]
    else:
        documents, labels = smoke_documents(), LABELS
        bundle = {
            "manifest": {"kind": "synthetic regression smoke; not a model ranking"}
        }
    texts = sorted({digest(d["text"]) for d in documents})
    if args.sweep and bundle["manifest"].get("split") == "test":
        parser.error("choose thresholds on development data, not the test split")
    selected, selection = {}, None
    if args.thresholds_from:
        source = json.loads(args.thresholds_from.read_text())
        if set(source["manifest"]["text_sha256"]) & set(texts):
            parser.error("threshold-selection documents overlap these documents")
        selected = source["selected_thresholds"]
        selection = dict(
            file=str(args.thresholds_from),
            file_sha256=hashlib.sha256(args.thresholds_from.read_bytes()).hexdigest(),
            dataset=source["manifest"]["dataset"],
            thresholds=selected,
        )
        missing = [
            name
            for name in args.models
            if MODELS[name]["adapter"] in THRESHOLDED and name not in selected
        ]
        if missing:
            parser.error(f"no development threshold for: {', '.join(missing)}")
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
    results, models, chosen = {}, {}, {}
    for name in args.models:
        thresholded = MODELS[name]["adapter"] in THRESHOLDED
        candidates = (
            args.sweep
            if args.sweep and thresholded
            else [selected.get(name, args.threshold)]
        )
        runs = {
            t: evaluate(
                documents,
                labels,
                lambda name=name, t=t: make_runner(
                    name, labels, args.download, args.device, t
                ),
                args.repeats,
                args.warmups,
            )
            for t in candidates
        }
        # Only a run that scored every document can choose a threshold; a model
        # that failed to load or rejected documents exports none.
        complete = [
            t for t, run in runs.items() if run["runs"][0]["metrics"]["headline"]
        ]
        best = max(
            complete or runs, key=lambda t: runs[t]["runs"][0]["metrics"]["micro"]["f1"]
        )
        if args.sweep and thresholded and complete:
            chosen[name] = best
        results[name] = runs[best]
        results[name]["micro_f1_ci95"] = f1_interval(
            documents, runs[best]["runs"][0]["records"], labels
        )
        if len(runs) > 1:
            results[name]["sweep"] = {
                str(t): run["runs"][0]["metrics"]["micro"] for t, run in runs.items()
            }
        models[name] = model_config(name, best)
    manifest = (
        dict(
            dataset=bundle["manifest"],
            document_ids=[d["document_id"] for d in documents],
            document_hashes={d["document_id"]: digest(d) for d in documents},
            text_sha256=texts,
            character_lengths=[len(d["text"]) for d in documents],
            labels=labels,
            label_count=len(labels),
            batch_size=1,
            device=args.device,
            repeats=args.repeats,
            warmups=args.warmups,
            truncation="reject, retain failed documents in denominator",
            threshold_selection=(
                "development sweep: " + ", ".join(map(str, args.sweep))
                if args.sweep
                else selection or f"fixed {args.threshold}"
            ),
            models=models,
        )
        | environment()
    )
    output = dict(manifest=manifest, results=results)
    if args.sweep:
        output["selected_thresholds"] = chosen
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(table(output))
    print(args.output)


if __name__ == "__main__":
    main()
