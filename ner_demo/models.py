"""Lazy adapters: imports never download weights or contact a paid service."""

import json
import re
import tempfile
from pathlib import Path

BACKBONES = {
    "gliner": ("microsoft/deberta-v3-base", "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"),
    "gliner25": (
        "microsoft/deberta-v3-small",
        "a36c739020e01763fe789b4b85e2df55d6180012",
    ),
}

PINS = {
    "gliner": (
        "urchade/gliner_medium-v2.1",
        "40ec419335d09393f298636f471328b722c6da9e",
    ),
    "gliner25": (
        "gliner-community/gliner_small-v2.5",
        "f227d3cd637bd4e6757ae143935316d062393341",
    ),
    "encoder": ("dslim/bert-base-NER", "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc"),
    "streaming": (
        "knowledgator/gliner-stream-pii-v1.0",
        "e871777dc4b3b688747a0433fff8d94a36fcc7b0",
    ),
}
RULES = {
    "person": ["John", "Zoë"],
    "organization": ["Microsoft"],
    "location": ["Paris"],
}


def rules(text, labels):
    return [
        {"start": m.start(), "end": m.end(), "text": m.group(), "label": label}
        for label in labels
        for phrase in RULES.get(label, [])
        for m in re.finditer(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text)
    ]


def load_gliner(name="gliner", download=False, device="cpu"):
    from gliner import GLiNER
    from huggingface_hub import snapshot_download

    model_id, revision = PINS[name]
    path = snapshot_download(
        model_id,
        revision=revision,
        local_files_only=not download,
        ignore_patterns=["*.fp16.safetensors", "*.bf16.safetensors"],
    )
    if name not in BACKBONES:
        return GLiNER.from_pretrained(
            path, local_files_only=not download, map_location=device
        ).eval()
    # Older checkpoints omit a tokenizer and/or nested backbone config. Stage pinned
    # companion files locally so GLiNER cannot resolve a floating backbone main.
    from transformers import AutoConfig, AutoTokenizer

    backbone = BACKBONES[name]
    kwargs = dict(revision=backbone[1], local_files_only=not download)
    tokenizer = (
        AutoTokenizer.from_pretrained(path, local_files_only=True)
        if (Path(path) / "tokenizer_config.json").exists()
        else AutoTokenizer.from_pretrained(backbone[0], **kwargs)
    )
    config = json.loads((Path(path) / "gliner_config.json").read_text())
    config["encoder_config"] = AutoConfig.from_pretrained(
        backbone[0], **kwargs
    ).to_dict()
    with tempfile.TemporaryDirectory(prefix="ner-gliner-") as temporary:
        staged = Path(temporary)
        tokenizer.save_pretrained(staged)
        (staged / "gliner_config.json").write_text(json.dumps(config))
        weights = (
            "model.safetensors"
            if (Path(path) / "model.safetensors").exists()
            else "pytorch_model.bin"
        )
        (staged / weights).symlink_to(Path(path) / weights)
        return GLiNER.from_pretrained(
            staged, local_files_only=True, map_location=device
        ).eval()


def check_gliner_length(model, text, labels):
    words = list(model.data_processor.words_splitter(text))
    if len(words) > model.config.max_len:
        raise ValueError("document exceeds GLiNER word limit")
    inputs, _ = model.data_processor.prepare_inputs([[w[0] for w in words]], [labels])
    tokenizer = model.data_processor.transformer_tokenizer
    ids = tokenizer(inputs[0], is_split_into_words=True, truncation=False)["input_ids"]
    # ponytail: conservative 512-subword ceiling for this encoder demo; windowing
    # needs its own boundary/overlap protocol before supporting longer inputs.
    if len(ids) > min(tokenizer.model_max_length, 512):
        raise ValueError("document plus labels exceeds conservative subword limit")
    return words


def gliner_predict(model, text, labels, threshold=0.5):
    check_gliner_length(model, text, labels)
    return model.predict_entities(text, labels, threshold=threshold, flat_ner=True)


def make_runner(name, labels, download=False, device="cpu", threshold=0.5):
    if name == "llm":
        from dotenv import load_dotenv
        from openai import OpenAI

        from ner_demo.core import digest
        from ner_demo.teacher import annotate

        load_dotenv()
        client = OpenAI(max_retries=0, timeout=60)
        return lambda text: annotate(
            client, {"document_id": digest(text), "text": text}, labels
        )
    if name == "rules":
        return lambda text: rules(text, labels)
    if name in {"gliner", "gliner25"}:
        model = load_gliner(name, download, device)
        return lambda text: gliner_predict(model, text, labels, threshold)
    if name == "encoder":
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )

        model_id, revision = PINS[name]
        kwargs = dict(revision=revision, local_files_only=not download)
        tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        model = AutoModelForTokenClassification.from_pretrained(model_id, **kwargs)
        pipe = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
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
                )
                for e in pipe(text)
                if mapping.get(e["entity_group"]) in labels
            ]

        return predict
    raise ValueError("unknown model")
