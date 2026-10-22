"""Lazy adapters: imports never download weights or contact a paid service.

Every adapter returns spans in one contract: Python character offsets
[start, end) into the input text, the requested label, the covered text, and
the model's own `score` when it has one. Adapters do not repair offsets; a span
that does not match the source reaches the scorer, which rejects it.
"""

import json
import re
import tempfile
import time
from pathlib import Path

from ner_demo.registry import MODELS

RULES = {
    "person": ["John", "Zoë"],
    "organization": ["Microsoft"],
    "location": ["Paris"],
}
# Weight formats no adapter here loads; skipping them keeps downloads smaller.
SKIP = [
    "*.fp16.safetensors",
    "*.bf16.safetensors",
    "*.h5",
    "*.msgpack",
    "*.onnx",
    "onnx/*",
    "openvino/*",
]


def rules(text, labels):
    return [
        {"start": m.start(), "end": m.end(), "text": m.group(), "label": label}
        for label in labels
        for phrase in RULES.get(label, [])
        for m in re.finditer(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text)
    ]


def snapshot(name, download=False):
    """Local directory of a pinned checkpoint. Downloads only when allowed."""
    from huggingface_hub import snapshot_download

    entry = MODELS[name]
    return Path(
        snapshot_download(
            entry["checkpoint"],
            revision=entry["revision"],
            local_files_only=not download,
            ignore_patterns=SKIP + entry.get("skip", []),
        )
    )


def load_gliner(name="gliner", download=False, device="cpu"):
    from gliner import GLiNER

    entry = MODELS[name]
    path = snapshot(name, download)
    # Everything below loads from local, pinned files only.
    kwargs = dict(local_files_only=True, map_location=device)
    if "labels_tokenizer" in entry:
        # Bi-encoder checkpoints omit the labels tokenizer; GLiNER would
        # otherwise load it from the Hub's floating main branch.
        from transformers import AutoTokenizer

        repo, revision = entry["labels_tokenizer"]
        kwargs["labels_tokenizer"] = AutoTokenizer.from_pretrained(
            repo, revision=revision, local_files_only=not download
        )
    if "backbone" not in entry:
        return GLiNER.from_pretrained(path, **kwargs).eval()
    # Older checkpoints omit a tokenizer and/or nested backbone config. Stage pinned
    # companion files locally so GLiNER cannot resolve a floating backbone main.
    from transformers import AutoConfig, AutoTokenizer

    repo, revision = entry["backbone"]
    pinned = dict(revision=revision, local_files_only=not download)
    tokenizer = (
        AutoTokenizer.from_pretrained(path, local_files_only=True)
        if (path / "tokenizer_config.json").exists()
        else AutoTokenizer.from_pretrained(repo, **pinned)
    )
    config = json.loads((path / "gliner_config.json").read_text())
    encoder = AutoConfig.from_pretrained(repo, **pinned).to_dict()
    # Checkpoints that saved a resized embedding (gliner_small-v2.5: 128003 rows)
    # must keep that size, or the backbone default (128100) fails to load.
    if config.get("vocab_size", -1) > 0:
        encoder["vocab_size"] = config["vocab_size"]
    config["encoder_config"] = encoder
    with tempfile.TemporaryDirectory(prefix="ner-gliner-") as temporary:
        staged = Path(temporary)
        tokenizer.save_pretrained(staged)
        (staged / "gliner_config.json").write_text(json.dumps(config))
        weights = (
            "model.safetensors"
            if (path / "model.safetensors").exists()
            else "pytorch_model.bin"
        )
        (staged / weights).symlink_to(path / weights)
        return GLiNER.from_pretrained(staged, **kwargs).eval()


def check_gliner_length(model, text, labels):
    """Reject inputs GLiNER would silently truncate (words or subwords)."""
    processor = model.data_processor
    words = list(processor.words_splitter(text))
    if len(words) > model.config.max_len:
        raise ValueError("document exceeds GLiNER word limit")
    tokens = [w[0] for w in words]
    labels_tokenizer = getattr(processor, "labels_tokenizer", None)
    if labels_tokenizer is None:
        # Uni-encoder: the label prompt shares the encoder window with the text.
        inputs, _ = processor.prepare_inputs([tokens], [labels])
        tokens = inputs[0]
    else:
        # Bi-encoder: labels have their own encoder, which truncates silently.
        lengths = labels_tokenizer(list(labels), truncation=False)["input_ids"]
        if max(map(len, lengths)) > labels_tokenizer.model_max_length:
            raise ValueError("label exceeds the label encoder's token limit")
    tokenizer = processor.transformer_tokenizer
    ids = tokenizer(tokens, is_split_into_words=True, truncation=False)["input_ids"]
    # ponytail: the encoder's position table caps the window (512 for DeBERTa-v3);
    # windowing needs its own boundary/overlap protocol before longer inputs.
    limit = min(
        tokenizer.model_max_length,
        model.config.encoder_config.max_position_embeddings,
    )
    if len(ids) > limit:
        raise ValueError("document plus labels exceeds the encoder token limit")
    return words


def gliner_predict(model, text, labels, threshold=0.5, embeddings=None):
    check_gliner_length(model, text, labels)
    if embeddings is None:
        return model.predict_entities(text, labels, threshold=threshold, flat_ner=True)
    return model.predict_with_embeds(
        text, embeddings, labels, threshold=threshold, flat_ner=True
    )


def load_gliner2(name="gliner2", download=False, device="cpu"):
    from gliner2 import AutoExtractor

    # Pass the pinned local snapshot: given a Hub ID, gliner2 2.1.0 loads the
    # tokenizer from main whatever `revision` says.
    return AutoExtractor.from_pretrained(
        str(snapshot(name, download)), map_location=device
    )


def gliner2_predict(model, text, labels, threshold=0.5):
    # gliner2 never truncates unless asked to, so the whole text is one window.
    words = list(model.processor.word_splitter(text))
    if len(words) > model.config.max_len:
        raise ValueError("document exceeds GLiNER2 word limit")
    # include_spans keeps repeated mentions; without it gliner2 deduplicates
    # entities by lower-cased text and returns no offsets.
    result = model.extract_entities(
        text, labels, threshold=threshold, include_spans=True, include_confidence=True
    )
    return [
        dict(
            start=e["start"],
            end=e["end"],
            text=e["text"],
            label=label,
            score=e["confidence"],
        )
        for label, found in result["entities"].items()
        for e in found
    ]


def make_runner(name, labels, download=False, device="cpu", threshold=0.5):
    entry = MODELS[name]
    adapter = entry["adapter"]
    if adapter == "openai":
        from dotenv import load_dotenv
        from openai import OpenAI

        from ner_demo.core import digest
        from ner_demo.teacher import annotate

        load_dotenv()
        client = OpenAI(max_retries=0, timeout=60)
        return lambda text: annotate(
            client, {"document_id": digest(text), "text": text}, labels
        )
    if adapter == "rules":
        return lambda text: rules(text, labels)
    if adapter in {"gliner", "gliner-bi"}:
        model = load_gliner(name, download, device)
        embeddings, started = None, time.perf_counter()
        if entry.get("label_cache"):
            embeddings = model.encode_labels(labels)

        def predict(text):
            return gliner_predict(model, text, labels, threshold, embeddings)

        # Reported separately from loading and from per-document latency.
        predict.label_cache_ms = (
            1000 * (time.perf_counter() - started) if embeddings is not None else None
        )
        return predict
    if adapter == "gliner2":
        model = load_gliner2(name, download, device)
        return lambda text: gliner2_predict(model, text, labels, threshold)
    if adapter == "token-classification":
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )

        kwargs = dict(revision=entry["revision"], local_files_only=not download)
        tokenizer = AutoTokenizer.from_pretrained(entry["checkpoint"], **kwargs)
        model = AutoModelForTokenClassification.from_pretrained(
            entry["checkpoint"], **kwargs
        )
        # "first": one tag per word, from its first sub-token (the BERT NER
        # setup). "simple" split words such as "Merkel" into "Merk" + "el".
        pipe = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="first",
            device=device,
        )
        mapping = {"PER": "person", "ORG": "organization", "LOC": "location"}
        if set(labels) - set(mapping.values()):
            raise ValueError("fixed encoder does not cover requested schema")

        def predict(text):
            if len(tokenizer.encode(text)) > model.config.max_position_embeddings:
                raise ValueError("document exceeds encoder token limit")
            return [
                dict(
                    start=int(e["start"]),
                    end=int(e["end"]),
                    text=text[e["start"] : e["end"]],
                    label=mapping[e["entity_group"]],
                    score=float(e["score"]),
                )
                for e in pipe(text)
                if mapping.get(e["entity_group"]) in labels
            ]

        return predict
    raise ValueError(f"{name} is not a general NER adapter")
