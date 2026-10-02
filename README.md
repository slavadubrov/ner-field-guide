# NER field guide

Companion code for the article
[NER Guide 2026: GLiNER, spaCy, Transformers, and LLMs](https://slavadubrov.com/blog/2026/04/02/ner-guide/).
It covers exact span scoring, local evaluation on a pinned dataset, conversion of
reviewed LLM annotations into GLiNER training input, ONNX FP32/INT8 checks, and a
streaming PII example.

The 32-document runs below check that each path works. They are too small to rank
models.

| Command group | Article section |
| --- | --- |
| Quickstart | GLiNER: span-to-label matching for open-vocabulary NER |
| Evaluation and dataset import | Evaluating NER: metrics, pitfalls, and test sets |
| Teacher annotations and training input | LLMs as teachers |
| ONNX packages and INT8 | Deployment optimization |
| Streaming PII | Streaming PII changes when you can release text |

## Setup and offline checks

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required.

```sh
uv sync --locked
uv run python -m scripts.run_all
uv run python -m unittest discover -s tests -v
uv run ruff check .
uv run ruff format --check .
uv build
```

The base package uses only the standard library. `run_all` runs the offline rules
smoke test. Imports never load models or call providers. The numbered files in
`scripts/` call the matching `ner_demo` module.

## Quickstart

```sh
uv sync --locked --all-extras
uv run --all-extras python -m ner_demo.quickstart --download   # first run
uv run --all-extras python -m ner_demo.quickstart              # cached models only
```

Example output:

```text
[{'start': 0, 'end': 10, 'text': 'Bill Gates', 'label': 'person', 'score': 0.99...},
 {'start': 19, 'end': 28, 'text': 'Microsoft', 'label': 'organization', 'score': 0.97...}]
```

Model downloads require explicit `--download`; checkpoints can use several GB.
Set `HF_HUB_OFFLINE=1` to force offline inference. `.env` is read only when you
select a paid adapter. Never commit credentials, weights, datasets, or generated
predictions. All output goes to the ignored `artifacts/` directory.

## Evaluation

```sh
uv run --all-extras python -m ner_demo.evaluate --models rules gliner --repeats 3
```

Without `--dataset`, this scores a few synthetic smoke documents. The rules adapter
is a tiny literal dictionary for known strings;
[spaCy EntityRuler](https://spacy.io/api/entityruler) is the native alternative in
a spaCy pipeline. It has no general-domain recall.

### Real dataset import and comparison

```sh
uv run --all-extras python -m ner_demo.data --split test --limit 32
uv run --all-extras python -m ner_demo.evaluate \
  --dataset artifacts/few-nerd-test.json --models rules encoder gliner gliner25 \
  --repeats 3 --output artifacts/few-nerd-evaluation.json
```

The first uncached model run needs `--download`. Example output on the first 32
test sentences (one line per adapter, shortened):

```text
rules    {"micro": {"tp": 0,  "fp": 0,  "fn": 54, "precision": 0.0,   "recall": 0.0,   "f1": 0.0}, ...}
encoder  {"micro": {"tp": 38, "fp": 42, "fn": 16, "precision": 0.475, "recall": 0.704, "f1": 0.567}, ...}
gliner   {"micro": {"tp": 39, "fp": 38, "fn": 15, "precision": 0.506, "recall": 0.722, "f1": 0.595}, ...}
gliner25 {"micro": {"tp": 44, "fp": 35, "fn": 10, "precision": 0.557, "recall": 0.815, "f1": 0.662}, ...}
```

The importer uses [Few-NERD](https://huggingface.co/datasets/DFKI-SLT/few-nerd),
config `supervised`, revision `205f3e9c9f3577ea2561d43f2f62dc249ab92d5b`: annotated
English Wikipedia sentences with eight coarse and 66 fine entity categories, under
CC-BY-SA-4.0. Keep attribution and share-alike terms when you redistribute derived
data. The command downloads one split's Parquet shard (100 MB cap) and keeps the
first N rows. It does not run remote dataset code.

To score more data, increase `--limit`. Use separate filenames and `--split train`
or `--split validation` for training and development data. Tokens are joined with
one ASCII space, so offsets refer to that sentence, not to the Wikipedia page. Only
the coarse person, organization, and location categories are kept. This is a
restricted task, not full Few-NERD scoring. Choose thresholds on development data
before a final comparison on test data.

Pinned candidates live in `ner_demo/models.py`:

| Adapter | Model and contract |
| --- | --- |
| `gliner` | `urchade/gliner_medium-v2.1`; tokenizer and backbone config pinned separately because the checkpoint omits them. |
| `gliner25` | `gliner-community/gliner_small-v2.5`, maintained GLiNER span model; same adapter. |
| `encoder` | [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER), fixed CoNLL labels; PER/ORG/LOC mapped, MISC dropped. |
| `llm` | Optional structured extractive annotations, with the returned model identity and usage. |

GLiNER v2.5 (`gliner-community`) and
[Fastino GLiNER2.5](https://huggingface.co/fastino/gliner2.5-base-v1) are different
model families. Fastino's model uses `gliner2.AutoExtractor`. Entity linking,
normalization, classification, and relations are outside span F1.

### Scoring rules

Each document has a unique `document_id`, `text`, and `entities`. An entity is:

```json
{"start": 0, "end": 4, "text": "John", "label": "person"}
```

* Offsets are Python Unicode character indices, half-open `[start,end)`.
  `text[start:end]` must equal the entity text.
* A match needs the same document ID, start, end, and label. Repeated mentions
  count separately. The scorer supports nested spans; the model comparison uses
  flat spans. Unknown labels and invalid spans cannot match.
* An unmatched prediction is a false positive; an unmatched gold span is a false
  negative. Duplicate predictions add false positives. Duplicate gold spans,
  duplicate result IDs, and extra result IDs reject the run.
* Missing, failed, and skipped documents stay in the gold denominator. Headline
  metrics are withheld unless every document was scored.
* Micro metrics sum counts. Macro metrics average over all requested labels,
  including labels with no gold spans.
* Batch size is one and the device is CPU by default. Threshold, labels, warmups,
  repetitions, package versions, checkpoint revisions, and raw predictions are
  saved. Failed calls do not enter p50/p95 latency. Inputs over GLiNER's word limit
  or 512 subwords (label prompt included) fail; nothing is truncated.

## Teacher annotations and training input

The teacher sends one request with a strict JSON Schema. It checks offsets, labels,
source text, and duplicates, and saves rejected annotations with reasons. It saves
the requested and returned model, prompt, settings, and usage. Records start
unreviewed.

```sh
cp .env.example .env
# Set OPENAI_API_KEY; optional OPENAI_BASE_URL changes the endpoint.
uv run --all-extras python -m ner_demo.teacher --allow-paid
```

The default model is `gpt-5.4-mini`, chosen for cost, not for NER accuracy. The
command makes at most one paid request, with no retries and at most 2048 output
tokens; see the provider's price page. It writes
`artifacts/teacher_annotations.jsonl`. Paid evaluation is also opt-in, capped at
32 documents of 4000 characters and one repetition:

```sh
uv run --all-extras python -m ner_demo.evaluate --models llm \
  --allow-paid --repeats 1 --warmups 0
```

Review the source and annotations, resolve rejected entries, and set
`reviewed: true`, a nonempty `reviewer`, `status: "ok"`, and a `split` (`train`,
`development`, or `test`) in each JSONL record. Keep `source_sha256` intact. Then
convert the reviewed file:

```sh
uv run --all-extras python -m ner_demo.training artifacts/reviewed.jsonl
```

This writes `artifacts/training/train.json`, `development.json`, and `test.json`
in GLiNER's training format (`tokenized_text` plus `ner` triples with inclusive
token ends). Every row passes through GLiNER's `SpanDataCollator`. Boundaries must
align with GLiNER's word splitter. Duplicate IDs, identical texts across splits,
unresolved rejections, and truncation fail. Pass the train and development files
to a GLiNER fine-tuning run and keep test for the final evaluation. The converter
does not train a model.

## ONNX packages and INT8

```sh
uv run --all-extras python -m ner_demo.onnx_check \
  --dataset artifacts/few-nerd-test.json --output artifacts/onnx-comparison
```

The output directory must be new; `--reuse` checks existing packages. The command
exports an FP32 package, quantizes a copy to INT8, checks each graph, config,
tokenizer, and external weights, and loads both through GLiNER's ONNX Runtime path.
PyTorch, FP32, and INT8 run on the same inputs, labels, threshold, and batch size.
`comparison.json` holds span differences, per-label recall and F1, latencies, and
package sizes. The command fails when FP32 spans differ from PyTorch, or when INT8
micro recall is more than 0.02 below PyTorch.

INT8 uses dynamic quantization of `MatMul`, `Gemm`, and the embedding `Gather`, and
keeps the twelve DeBERTa FFN down-projections (`layer.N/output/dense`) in FP32.
Quantizing only those twelve layers already makes the model return no entities. Example output on the
32 test sentences (Apple M5 Max, ONNX Runtime 1.24.4):

```text
{"package_bytes": {"fp32": 790362476, "int8": 353299069},
 "micro_recall": {"pytorch": 0.722, "fp32": 0.722, "int8": 0.722},
 "differing_documents": {"fp32": 0, "int8": 4}}
```

| Path | Package | p50 latency | Micro precision | Micro recall |
| --- | --- | --- | --- | --- |
| PyTorch | - | 64-66 ms | 0.506 | 0.722 |
| ONNX FP32 | 790 MB | 16-19 ms | 0.506 | 0.722 |
| ONNX INT8 | 353 MB | 13-15 ms | 0.487 | 0.722 |

INT8 changed the spans in 4 of 32 documents and added 3 false positives. Check
per-label recall on your own data before you deploy it. See
[ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

## Streaming PII

[GLiNER Streaming PII](https://huggingface.co/knowledgator/gliner-stream-pii-v1.0)
has a session API that returns offsets into the accumulated text. Its checkpoint is
pinned separately. This demo keeps labels fixed and gives each session its own ID.

```sh
uv run --all-extras python -m ner_demo.streaming --download
```

`BufferedPII` holds all text until a final full-text detection pass, then releases
it with detected characters masked and clears the session. Exit, exceptions,
cancellation, and overflow also clear buffered text and model state. The
16,000-character cap limits input storage, not model memory. Release delay
includes the whole input and the final inference. A shorter delay needs its own
tested policy: a fixed lookbehind cannot protect arbitrarily long entities.
Detector misses can still leak PII. The tests check buffering behavior, not
privacy recall. Do not mix character-level leakage metrics with document NER F1.

## Optional integration checks

```sh
NER_LOCAL_CHECKS=1 HF_HUB_OFFLINE=1 uv run --all-extras python -m unittest discover -s tests -v
```

These checks load cached GLiNER weights and test collator alignment and truncation
rejection. Normal tests use synthetic fixtures and a mock HTTP transport through
the real OpenAI SDK. Downloads and paid inference never run in normal tests.
