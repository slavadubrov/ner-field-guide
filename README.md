# NER field guide

A small companion project for a technical article: exact span scoring, reproducible
local evaluation, reviewed annotation conversion, ONNX checks, and a separate
streaming PII example. Synthetic smoke results are regression checks, **not a model
ranking**. No training run or universal quality/speed advantage is claimed.

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

The base package uses the standard library. `run_all` now runs only the offline
rules smoke. Imports never load models or call providers. Optional dependencies:

```sh
uv sync --locked --all-extras
uv run --all-extras python -m ner_demo.quickstart        # cached models only
uv run --all-extras python -m ner_demo.quickstart --download
```

Model downloads require explicit `--download`; checkpoints can occupy several GB.
Use `HF_HUB_OFFLINE=1` to enforce fully offline local inference. `.env` is read only
when an explicitly selected paid adapter is initialized. Never commit credentials,
weights, datasets, or generated predictions. All example output goes to ignored
`artifacts/`. Each old numbered script delegates to the corresponding module;
03 and 05 share the same structured teacher implementation.

## Exact evaluation protocol

Each document has a unique `document_id`, `text`, and `entities`. An entity is:

```json
{"start": 0, "end": 4, "text": "John", "label": "person"}
```

Offsets are Python Unicode character indices, **half-open `[start,end)`**, not
UTF-8 bytes or JavaScript UTF-16 units. `text[start:end]` must equal entity text.
Matching uses document ID, start, end and label, case sensitively. Repeated mentions
remain separate. Overlapping/nested spans are supported by the scorer; the common
model comparison uses flat spans. Unknown labels and invalid spans cannot match.

* An unmatched prediction is FP; unmatched gold is FN. Duplicate predictions add
  FP. Duplicate gold, duplicate result IDs and extra result IDs reject the run.
* Missing, failed and skipped documents remain in the gold denominator. Partial
  predictions on failed calls can add FP but cannot add TP. An unknown-label span
  adds a micro FP and is counted separately from known-label metrics.
* Micro metrics sum occurrence counts. Macro precision/recall/F1 average each
  metric over **all requested labels**, including zero-support labels. Undefined
  precision/recall/F1 are zero. Per-label support is gold TP+FN.
* Completion = successful calls / planned documents; scoring coverage = complete
  documents with valid spans / planned documents, separately per repetition.
  Headline metrics are withheld unless coverage is 100%. Diagnostic micro metrics
  still include every gold document; never compare incomplete runs as rankings.
* One record is written for every model/document/repetition, including loading
  errors. Failed timings do not enter successful-call p50/p95; p95 uses nearest
  rank. Initialization and warmup failures are reported separately.
* Batch size is fixed at one, CPU is the default, and threshold, labels, lengths,
  warmups, repetitions, package versions, source hash, checkpoint revisions and
  raw predictions are recorded. GLiNER inputs exceeding its word limit or this
  demo's conservative 512-subword limit **including the real label prompt** fail
  explicitly. The fixed encoder also rejects overlength inputs. No hidden window
  merging or truncation is used. Timings include adapter validation and decoding.

```sh
uv run --all-extras python -m ner_demo.evaluate --models rules gliner --repeats 3
```

Rules are a deliberately tiny literal dictionary, useful for stable known strings;
[spaCy EntityRuler](https://spacy.io/api/entityruler) is the native alternative when
an application already uses spaCy. This dictionary has no expected general-domain
recall. It is fixed independently of the real evaluation split.

## Real dataset import and comparison

The importer uses [Few-NERD](https://huggingface.co/datasets/DFKI-SLT/few-nerd),
`supervised`, revision `205f3e9c9f3577ea2561d43f2f62dc249ab92d5b`.
It contains annotated English Wikipedia sentences, with train/validation/test
splits, eight coarse entity categories and 66 fine categories. The published
license is CC-BY-SA-4.0 and the chosen shard has no authentication gate. Preserve
attribution and share-alike obligations when redistributing derived data. The
import command downloads one split's Parquet shard (100 MB hard cap), then keeps
only the first N rows. It does not execute remote dataset code.

```sh
uv run --all-extras python -m ner_demo.data --split test --limit 32
uv run --all-extras python -m ner_demo.evaluate \
  --dataset artifacts/few-nerd-test.json --models rules encoder gliner gliner25 \
  --repeats 3 --output artifacts/few-nerd-evaluation.json
```

The first uncached model run needs `--download`. To scale the sample, increase
`--limit`. Use separate filenames and `--split train` / `--split validation` for
training and development. IDs include dataset/config/split/source ID. The bundle
retains source and document hashes, label mapping and deterministic selection.
Tokens are joined with one ASCII space: offsets refer to that reconstructed
sentence, **not the original Wikipedia page**. Contiguous equal fine tags define
mentions before projecting coarse person/organization/location; other categories
are excluded. This is an explicitly restricted task, not full Few-NERD scoring.

The fixed encoder is [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER),
with PER/ORG/LOC mapped to the common schema and MISC excluded. Its CoNLL annotation
conventions do not perfectly match Few-NERD. Schema compatibility here is a
practical projection, not evidence of identical annotation policies. CoNLL-2003
is a useful historical alternative but has source-text licensing/access concerns;
this project does not redistribute it. Few-NERD sentence IDs do not establish
page-level independence. Neither dataset choice nor revision pinning proves
absence of pretraining contamination. Freeze selection/thresholds on development
before an independent final comparison; the small test command is an execution
check, not a representative final benchmark.

Pinned candidates live in `ner_demo/models.py`:

| Adapter | Family and contract |
| --- | --- |
| `gliner` | Retained `urchade/gliner_medium-v2.1`; tokenizer/backbone metadata separately pinned because the original package omitted them. |
| `gliner25` | `gliner-community/gliner_small-v2.5`, maintained GLiNER span model; same extraction adapter. |
| `encoder` | Fixed CoNLL-trained BERT; supports only its trained label inventory. |
| `llm` | Optional structured extractive annotations with returned identity and usage. |

Currentness checked 6 September 2026: [GLiNER's maintained API](https://github.com/urchade/GLiNER)
and [GLiNER v2.5 card](https://huggingface.co/gliner-community/gliner_small-v2.5)
are separate from [Fastino GLiNER2.5](https://huggingface.co/fastino/gliner2.5-base-v1),
which uses `gliner2.AutoExtractor` and supports additional structured tasks.
This project does not transfer Fastino interfaces or published results to GLiNER.
Entity linking, normalization, classification and relations are outside span F1.

## Teacher annotations and training input

The teacher uses strict JSON Schema, validates offsets, labels, source text,
duplicates and generation completion, and saves rejected annotations with reasons.
It never reconstructs offsets using substring matching and never silently falls
back to GLiNER. Requested/actual producer and model, exact prompt, settings, finish
status, returned response ID and usage are retained. A provider error stays an
error. Valid generation alone does not authorize training: records start unreviewed.

The default `gpt-5.4-mini` remains an available, inexpensive structured-output
candidate: the [official model page](https://developers.openai.com/api/docs/models/gpt-5.4-mini)
lists standard input/output prices of $0.75/$4.50 per million tokens (checked
2026-09-06). This is a cost-based demo choice, not a claim of superior NER accuracy.
The alias may change; returned model identity is saved, with no invented immutable
revision. Custom model pricing is left unspecified. Account access and live
extraction quality require a separate paid check.

```sh
cp .env.example .env
# Set OPENAI_API_KEY locally; optional OPENAI_BASE_URL changes the endpoint.
uv run --all-extras python -m ner_demo.teacher --allow-paid
```

This command makes at most one request, with no retries and 2048 maximum output
tokens. It describes NVIDIA's **proposed** Arm acquisition in 2020. Paid evaluation
is also opt-in, capped at 32 documents of 4000 characters and one repetition:

```sh
uv run --all-extras python -m ner_demo.evaluate --models llm \
  --allow-paid --repeats 1 --warmups 0
```

Review the complete source and annotations, resolve rejected entries, and set
`reviewed: true`, a nonempty `reviewer`, `status: "ok"`, and an explicit `split`
(`train`, `development`, `test`) in each JSONL record. Keep `source_sha256` intact.
Review metadata is an explicit human attestation, not an automated quality proof.

```sh
uv run --all-extras python -m ner_demo.training artifacts/reviewed.jsonl
```

Conversion uses the selected GLiNER word splitter; boundaries must align exactly
and fit its span width. Output is `tokenized_text` plus `ner` triples with inclusive
**token** ends, the input contract of GLiNER's `SpanDataCollator`. Every row passes
through that real collator and every positive span/class position is checked.
Duplicate document IDs, cross-split identical texts, unresolved rejection and
truncation fail. The CLI requires all three nonempty splits and writes separate
JSON files. Only train/development should be passed to a later training run;
reserve test for final evaluation. **No optimizer steps or fine-tuning run are
performed by this converter. No trained student model is produced.**

## ONNX packages and parity

```sh
uv run --all-extras python -m ner_demo.onnx_check \
  --dataset artifacts/few-nerd-test.json --output artifacts/onnx-comparison
```

The output directory must be new; `--reuse` checks existing packages. The checker
requires the graph, config, loadable tokenizer and referenced external weights,
checks the graph, and loads it through GLiNER's real ONNX Runtime path. PyTorch,
FP32 and INT8 use identical inputs, labels, threshold, batch size, warmups and
repetitions. The result includes exact-span disagreements, per-label recall/F1,
latencies and complete package bytes. Any disagreement returns a nonzero exit
status after writing the comparison. INT8 is allowed to lose accuracy; the loss
must be visible before deployment. No promised speedups are multiplied together.
The pinned ORT quantizer can order a node before values captured by its `If`
subgraphs; the exporter repairs dependency order with the standard-library
`TopologicalSorter`, then revalidates the graph. This changes ordering, not math.

On the local 32-document check, FP32 preserved all PyTorch spans; the generated
INT8 model returned no entities (recall 0). It is **not a validated deployment
artifact**. The parity gate fails and saves the evidence. Quantization-aware
training or a separately validated quantization configuration is future work;
smaller files alone do not establish usability.
See [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

## Streaming PII is a separate task

[GLiNER Streaming PII](https://huggingface.co/knowledgator/gliner-stream-pii-v1.0)
publishes a session API supported by pinned GLiNER 0.2.28. The checkpoint is pinned
independently. Incremental snapshots use accumulated-document offsets and chunks
are concatenated verbatim. This demo keeps labels fixed and isolates session IDs.

```sh
uv run --all-extras python -m ner_demo.streaming --download
```

`BufferedPII` withholds **all text** until a final stateless full-text detection
pass, then masks detected characters and clears the session. This also flushes
partial identifiers without inventing an empty-chunk recompute API. Context exit,
exception, cancellation and overflow clear buffered source and model session state.
The 16,000-character cap bounds input storage, not model memory. Release delay
includes the entire input duration and final inference. Lower release latency
needs a separately validated policy; a finite lookbehind cannot guarantee
protection for arbitrarily long entities. Detector misses can still leak PII at
final release. Contract tests establish buffering behavior, not privacy recall.
Do not mix character masking/release leakage metrics with document NER F1.

## Optional integration checks

```sh
NER_LOCAL_CHECKS=1 HF_HUB_OFFLINE=1 uv run --all-extras python -m unittest discover -s tests -v
```

These checks load only cached retained GLiNER weights and validate actual collator
alignment and truncation rejection. Normal tests use synthetic fixtures and a
mock HTTP transport through the real OpenAI SDK when the teacher extra is present.
Large downloads and paid inference never run in ordinary tests or package imports.
