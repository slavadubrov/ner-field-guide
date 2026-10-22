"""Pinned model registry. Data only: importing it loads nothing.

`status` says how far each entry was checked in this repository:
  validated  - loaded and run with the real pinned weights (see docs/models.md)
  contract   - adapter covered by offline contract tests with fakes only
  implemented - adapter code exists without tests
  blocked    - not integrated; `limitations` names the blocker
`verified` is the date the checkpoint, revision and license were checked
against the official sources listed in `sources`.
"""

VERIFIED = "2026-10-22"

BI_ENCODER = dict(
    family="GLiNER bi-encoder v2.0 (Knowledgator)",
    role="current",
    checkpoint="knowledgator/gliner-bi-small-v2.0",
    revision="3d74c1bf459b8b1c0be1ecbddd679416ce005418",
    # The checkpoint omits the tokenizer of its label encoder.
    labels_tokenizer=(
        "sentence-transformers/all-MiniLM-L12-v2",
        "a50ef00143b4d5391434df20ae11632588ac25be",
    ),
    adapter="gliner-bi",
    labels="open",
    languages="en",
    task="NER",
    license="Apache-2.0",
    extra="models",
    sources=[
        "https://huggingface.co/knowledgator/gliner-bi-small-v2.0",
        "https://arxiv.org/abs/2602.18487",
    ],
    verified=VERIFIED,
    status="validated",
    limitations="Encodes the labels on every call.",
)

MODELS = {
    "rules": dict(
        family="Literal dictionary",
        role="smoke test",
        checkpoint=None,
        revision=None,
        adapter="rules",
        labels="fixed",
        languages="any",
        task="NER",
        license="MIT (this repository)",
        extra=None,
        sources=[],
        verified=VERIFIED,
        status="validated",
        limitations="Matches four known strings; no general recall.",
    ),
    "gliner": dict(
        family="GLiNER v2.1 (urchade)",
        role="baseline",
        checkpoint="urchade/gliner_medium-v2.1",
        revision="40ec419335d09393f298636f471328b722c6da9e",
        # The checkpoint omits its tokenizer and nested backbone config.
        backbone=(
            "microsoft/deberta-v3-base",
            "8ccc9b6f36199bec6961081d44eb72fb3f7353f3",
        ),
        adapter="gliner",
        labels="open",
        languages="en",
        task="NER",
        license="Apache-2.0",
        extra="models",
        sources=["https://huggingface.co/urchade/gliner_medium-v2.1"],
        verified=VERIFIED,
        status="validated",
        limitations="Article reproducibility baseline; only model with a "
        "validated ONNX FP32/INT8 path.",
    ),
    "gliner25": dict(
        family="GLiNER v2.5 (gliner-community)",
        role="current",
        checkpoint="gliner-community/gliner_small-v2.5",
        revision="f227d3cd637bd4e6757ae143935316d062393341",
        backbone=(
            "microsoft/deberta-v3-small",
            "a36c739020e01763fe789b4b85e2df55d6180012",
        ),
        adapter="gliner",
        labels="open",
        languages="en",
        task="NER",
        license="Apache-2.0",
        extra="models",
        sources=["https://huggingface.co/gliner-community/gliner_small-v2.5"],
        verified=VERIFIED,
        status="validated",
        limitations="Model generation v2.5, unrelated to Fastino GLiNER2.5.",
    ),
    "gliner-bi": BI_ENCODER,
    # Same checkpoint; labels are encoded once per run (timed separately as
    # `label_cache_ms`) and reused for every document.
    "gliner-bi-cached": dict(
        BI_ENCODER,
        label_cache=True,
        limitations="Label embeddings computed once per run and reused.",
    ),
    "gliner-multi": dict(
        family="GLiNER v2.1 multilingual (urchade)",
        role="current",
        checkpoint="urchade/gliner_multi-v2.1",
        revision="443d26d654e0324125a96bebd8e796c14ff2efe6",
        backbone=(
            "microsoft/mdeberta-v3-base",
            "a0484667b22365f84929a935b5e50a51f71f159d",
        ),
        skip=["pytorch_model.bin"],  # duplicate of model.safetensors
        adapter="gliner",
        labels="open",
        languages="multilingual (card lists no languages; mDeBERTa-v3 backbone)",
        task="NER",
        license="Apache-2.0",
        extra="models",
        sources=["https://huggingface.co/urchade/gliner_multi-v2.1"],
        verified=VERIFIED,
        status="validated",
        limitations="No per-language accuracy published for v2.1; the "
        "English/German/Russian examples here are functional checks only.",
    ),
    "gliner2": dict(
        family="GLiNER2.5 (Fastino)",
        role="current",
        checkpoint="fastino/gliner2.5-base-v1",
        revision="54785d51df8d86a0d8f2eb212fec15871aad2c4d",
        adapter="gliner2",
        labels="open",
        languages="en",
        task="NER",
        license="Apache-2.0",
        extra="gliner2",
        sources=[
            "https://huggingface.co/fastino/gliner2.5-base-v1",
            "https://github.com/fastino-ai/GLiNER2/releases/tag/v2.1.0",
        ],
        verified=VERIFIED,
        status="validated",
        limitations="Separate `gliner2` package. Overlaps are resolved per label, "
        "so spans with different labels may overlap. NER only here.",
    ),
    "encoder": dict(
        family="BERT token classification (CoNLL-2003)",
        role="baseline",
        checkpoint="dslim/bert-base-NER",
        revision="d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc",
        adapter="token-classification",
        labels="fixed",
        languages="en",
        task="NER",
        license="MIT",
        extra="models",
        sources=["https://huggingface.co/dslim/bert-base-NER"],
        verified=VERIFIED,
        status="validated",
        limitations="Fixed PER/ORG/LOC/MISC labels; MISC is dropped. "
        "Older supervised baseline, not a current recommendation.",
    ),
    "llm": dict(
        family="OpenAI structured outputs (paid, opt-in)",
        role="baseline",
        # Dated API snapshot, not the moving alias; each call also saves the
        # model ID the API reports.
        checkpoint="gpt-5.4-mini-2026-03-17",
        revision=None,
        adapter="openai",
        labels="open",
        languages="multilingual (provider claim)",
        task="NER",
        license="OpenAI terms of use",
        extra="teacher",
        sources=["https://developers.openai.com/api/docs/models/gpt-5.4-mini"],
        verified=VERIFIED,
        status="contract",
        limitations="Paid; not run here. Offline tests use a mock HTTP transport.",
    ),
    "streaming": dict(
        family="GLiNER Streaming PII (Knowledgator)",
        role="current",
        checkpoint="knowledgator/gliner-stream-pii-v1.0",
        revision="e871777dc4b3b688747a0433fff8d94a36fcc7b0",
        adapter="gliner-stream",
        labels="open",
        languages="en",
        task="PII (streaming)",
        license="Apache-2.0",
        extra="models",
        sources=["https://huggingface.co/knowledgator/gliner-stream-pii-v1.0"],
        verified=VERIFIED,
        status="validated",
        limitations="Buffering demo only; detector recall is not measured here.",
    ),
}


def command(name):
    entry = MODELS[name]
    uv = f"uv run --extra {entry['extra']}" if entry["extra"] else "uv run"
    if entry["adapter"] == "rules":
        return f"{uv} python -m ner_demo.evaluate"
    if entry["adapter"] == "openai":
        return (
            f"{uv} python -m ner_demo.evaluate --models llm --allow-paid "
            "--repeats 1 --warmups 0"
        )
    if entry["adapter"] == "gliner-stream":
        return f"{uv} python -m ner_demo.streaming --download"
    return f"{uv} python -m ner_demo.quickstart --model {name} --examples --download"


def table():
    """Markdown model-support table; README.md embeds this output verbatim."""
    lines = [
        "| Run name | Family (role) | Checkpoint @ revision | Languages; task "
        "| Extra | Command | Status | Limitations |",
        "|" + " --- |" * 8,
    ]
    for name, entry in MODELS.items():
        checkpoint = f"`{entry['checkpoint']}`" if entry["checkpoint"] else "-"
        if entry["revision"]:
            checkpoint += f" @ `{entry['revision']}`"
        lines.append(
            f"| `{name}` | {entry['family']} ({entry['role']}) | {checkpoint} "
            f"| {entry['languages']}; {entry['task']} | {entry['extra'] or '-'} "
            f"| `{command(name)}` | {entry['status']} | {entry['limitations']} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(table())
