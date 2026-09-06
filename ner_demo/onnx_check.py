"""Export complete packages and measure PyTorch/ONNX FP32/INT8 separately."""

import argparse
import importlib.metadata
import json
import platform
import shutil
import time
from pathlib import Path

from ner_demo.core import LABELS, digest, smoke_documents
from ner_demo.evaluate import evaluate
from ner_demo.models import PINS, gliner_predict, load_gliner


def validate_package(path):
    import onnx
    from transformers import AutoTokenizer

    path = Path(path)
    for name in (
        "model.onnx",
        "gliner_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ):
        if not (path / name).is_file():
            raise ValueError(f"incomplete ONNX package: missing {name}")
    graph = onnx.load(str(path / "model.onnx"), load_external_data=False)
    for tensor in onnx.external_data_helper._get_all_tensors(graph):
        for item in tensor.external_data:
            if item.key == "location":
                external = (path / item.value).resolve()
                if (
                    not external.is_relative_to(path.resolve())
                    or not external.is_file()
                ):
                    raise ValueError("missing or unsafe ONNX external weights")
    onnx.checker.check_model(str(path / "model.onnx"))
    AutoTokenizer.from_pretrained(path, local_files_only=True)
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def sort_graph(graph):
    """Include subgraph captures when ordering nodes (ORT 1.24 quantizer omits them)."""
    from graphlib import TopologicalSorter

    from onnx import AttributeProto

    dependencies = []
    local = {value.name for value in graph.input} | {
        value.name for value in graph.initializer
    }
    producers = {name: i for i, node in enumerate(graph.node) for name in node.output}
    for node in graph.node:
        inputs = set(node.input) - {""}
        for attribute in node.attribute:
            subgraphs = (
                [attribute.g]
                if attribute.type == AttributeProto.GRAPH
                else attribute.graphs
                if attribute.type == AttributeProto.GRAPHS
                else []
            )
            for subgraph in subgraphs:
                inputs.update(sort_graph(subgraph))
        dependencies.append(inputs)
    order = TopologicalSorter(
        {
            i: {producers[name] for name in names if name in producers}
            for i, names in enumerate(dependencies)
        }
    ).static_order()
    nodes = [graph.node[i] for i in order]
    del graph.node[:]
    graph.node.extend(nodes)
    return (
        set().union(*dependencies, {value.name for value in graph.output})
        - local
        - producers.keys()
    )


def quantize_package(fp32, int8):
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(fp32 / "model.onnx"),
        str(int8 / "model.onnx"),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    graph = onnx.load(str(int8 / "model.onnx"))
    sort_graph(graph.graph)
    onnx.save(graph, str(int8 / "model.onnx"))
    validate_package(int8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/onnx-comparison")
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument(
        "--reuse", action="store_true", help="Validate existing complete packages"
    )
    args = parser.parse_args()
    from gliner import GLiNER

    initialized = time.perf_counter()
    model = load_gliner()
    initialization_ms = 1000 * (time.perf_counter() - initialized)
    fp32, int8 = args.output / "fp32", args.output / "int8"
    if not args.reuse:
        if args.output.exists():
            raise ValueError("output exists; use a new directory or --reuse")
        args.output.mkdir(parents=True)
        model.export_to_onnx(str(fp32))
        validate_package(fp32)
        shutil.copytree(fp32, int8)
        quantize_package(fp32, int8)
    sizes = {
        name: validate_package(path) for name, path in [("fp32", fp32), ("int8", int8)]
    }
    docs, labels = smoke_documents(), LABELS
    dataset_manifest = {"kind": "synthetic smoke only"}
    if args.dataset:
        bundle = json.loads(args.dataset.read_text())
        docs, labels = bundle["documents"], bundle["manifest"]["labels"]
        if digest(docs) != bundle["manifest"]["documents_sha256"]:
            raise ValueError("dataset hash mismatch")
        dataset_manifest = bundle["manifest"]
    results = {
        "pytorch": evaluate(
            docs, labels, lambda: lambda text: gliner_predict(model, text, labels)
        )
    }
    for name, path in [("fp32", fp32), ("int8", int8)]:

        def factory(path=path):
            runtime = GLiNER.from_pretrained(
                str(path),
                load_onnx_model=True,
                local_files_only=True,
                map_location="cpu",
            )
            return lambda text: gliner_predict(runtime, text, labels)

        results[name] = evaluate(docs, labels, factory)
    results["pytorch"]["initialization_ms"] = initialization_ms

    def spans(record):
        return sorted((e["start"], e["end"], e["label"]) for e in record["entities"])

    parity = {}
    for name in ("fp32", "int8"):
        differences = set()
        for run in results[name]["runs"]:
            reference = results["pytorch"]["runs"][run["repeat"]]["records"]
            for i, row in enumerate(run["records"]):
                if (
                    row["status"] != "ok"
                    or reference[i]["status"] != "ok"
                    or spans(row) != spans(reference[i])
                ):
                    differences.add(row["document_id"])
        parity[name] = sorted(differences)
    report = dict(
        model=PINS["gliner"],
        dataset=dataset_manifest,
        documents_sha256=digest(docs),
        labels=labels,
        device="cpu",
        batch_size=1,
        threshold=0.5,
        warmups=1,
        repeats=3,
        package_bytes=sizes,
        platform=platform.platform(),
        packages={
            name: importlib.metadata.version(name)
            for name in ["gliner", "torch", "transformers", "onnx", "onnxruntime"]
        },
        source_sha256=digest(
            {p.name: p.read_text() for p in Path(__file__).parent.glob("*.py")}
        ),
        differing_documents=parity,
        results=results,
    )
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(package_bytes=sizes, differing_documents=parity)))
    if any(parity.values()):
        raise SystemExit(
            "ONNX differences detected; inspect comparison.json per-label metrics"
        )


if __name__ == "__main__":
    main()
