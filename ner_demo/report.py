"""Print Markdown tables for a saved evaluation JSON file."""

import argparse
import json
from pathlib import Path


def number(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def table(output):
    manifest, results = output["manifest"], output["results"]
    dataset = manifest["dataset"]
    source = (
        f"{dataset['dataset']}@{dataset['revision'][:7]} split={dataset['split']}"
        if {"dataset", "revision", "split"} <= dataset.keys()
        else dataset.get("kind", "custom documents")
    )
    git = (manifest.get("git_commit") or "unknown")[:7]
    thresholds = manifest["threshold_selection"]
    if isinstance(thresholds, dict):  # chosen by a --sweep run on other data
        chosen_on = thresholds["dataset"]
        thresholds = (
            f"from {thresholds['file']}, chosen on "
            f"{chosen_on.get('dataset', 'custom documents')} "
            f"split={chosen_on.get('split', '?')}"
        )
    lines = [
        f"Data: {source}; {len(manifest['document_ids'])} documents; "
        f"labels: {', '.join(manifest['labels'])}",
        f"Run: {manifest.get('cpu') or manifest['platform']}, "
        f"{manifest.get('logical_cpus')} logical CPUs, "
        f"torch threads {manifest.get('torch_threads', '-')}, "
        f"device {manifest['device']}, batch {manifest['batch_size']}, "
        f"{manifest['warmups']} warmup, {manifest['repeats']} repeats; "
        f"git {git}{' (dirty)' if manifest.get('git_dirty') else ''}",
        f"Thresholds: {thresholds}",
        "",
        "| Model | Checkpoint | Labels | Threshold | P | R | F1 | F1 95% CI | TP/FP/FN "
        "| Attempted | Scored | Failed | Missing | Load ms | Label cache ms "
        "| First call ms | p50 ms | p95 ms |",
        "|" + " --- |" * 18,
    ]
    per_label = []
    for name, result in results.items():
        config = manifest["models"][name]
        metrics = result["runs"][0]["metrics"]
        micro = metrics["micro"]
        checkpoint = config["checkpoint"] or "-"
        if config["revision"]:
            checkpoint += f"@{config['revision'][:7]}"
        flag = (
            " (varies by repeat)" if result.get("metrics_vary_across_repeats") else ""
        )
        if result.get("initialization_error"):
            flag += f" (load failed: {result['initialization_error']})"
        interval = result.get("micro_f1_ci95")
        interval = "-" if interval is None else "-".join(map(number, interval))
        lines.append(
            f"| {name}{flag} | {checkpoint} | {config['labels']} "
            f"| {number(config['threshold'], 2)} "
            f"| {number(micro['precision'])} | {number(micro['recall'])} "
            f"| {number(micro['f1'])} | {interval} "
            f"| {micro['tp']}/{micro['fp']}/{micro['fn']} "
            f"| {metrics['attempted_documents']} | {metrics['valid_documents']} "
            f"| {metrics['failed_documents']} | {metrics['missing_documents']} "
            f"| {number(result['initialization_ms'], 0)} "
            f"| {number(result.get('label_cache_ms'), 0)} "
            f"| {number(result['warmup_ms'][0] if result.get('warmup_ms') else None, 1)} "
            f"| {number(result['latency_ms']['p50'], 1)} "
            f"| {number(result['latency_ms']['p95'], 1)} |"
        )
        per_label.append(
            f"| {name} | "
            + " | ".join(
                number(metrics["per_label"][label]["f1"])
                for label in manifest["labels"]
            )
            + " |"
        )
    lines += [
        "",
        "| Per-label F1 | " + " | ".join(manifest["labels"]) + " |",
        "|" + " --- |" * (len(manifest["labels"]) + 1),
        *per_label,
    ]
    sweeps = {name: r["sweep"] for name, r in results.items() if "sweep" in r}
    if sweeps:
        lines += [
            "",
            "| Sweep (micro F1) | threshold: F1 | Selected |",
            "| --- | --- | --- |",
        ]
        for name, sweep in sweeps.items():
            selected = output.get("selected_thresholds", {}).get(name)
            if selected is None:
                lines.append(f"| {name} | no threshold scored every document | none |")
                continue
            grid = [float(t) for t in sweep]
            # A best value at the end of the grid may not be the best threshold.
            edge = (
                " (grid edge: widen the sweep)"
                if selected in (min(grid), max(grid))
                else ""
            )
            lines.append(
                f"| {name} | "
                + ", ".join(f"{t}: {number(m['f1'])}" for t, m in sweep.items())
                + f" | {selected}{edge} |"
            )
    lines.append(
        "\nScored = completed with only valid spans. Failed and missing documents "
        "stay in the denominator. F1 95% CI = document bootstrap, 1,000 resamples, "
        "the same resamples for every model. First call = warmup on the first "
        "document."
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    print(table(json.loads(parser.parse_args().evaluation.read_text())))


if __name__ == "__main__":
    main()
