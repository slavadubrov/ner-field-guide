# NER field guide

Companion code for the article
[NER Guide 2026: GLiNER, spaCy, Transformers, and LLMs](https://slavadubrov.com/blog/2026/04/02/ner-guide/).

A small reference project for engineers who need to choose or check a named
entity recognition (NER) model. It runs pinned GLiNER-family checkpoints and two
baselines on the same pinned documents and scores them as exact character spans,
with failed and missing documents kept in the denominator. It also contains the
article's other workflows: reviewed LLM annotations as GLiNER training input,
ONNX FP32/INT8 export checks, and buffered streaming PII redaction.

It is not a leaderboard. The runs below check that each path works under one set
of rules on one machine; they do not rank model families.

The code follows these article sections: GLiNER, its bi-encoder, and GLiNER2.5
(quickstart and adapters); evaluating NER (scorer and evaluation runner); LLMs as
teachers (annotation and training input); deployment optimization (ONNX); and
streaming PII.

## Install and run the offline example

Python 3.11-3.14 and [uv](https://docs.astral.sh/uv/). The base package uses only
the standard library.

```sh
uv sync --locked
uv run python -m scripts.run_all
```

This scores a literal-dictionary adapter on three synthetic documents (repeated
mentions, non-ASCII text, an empty document) and prints the report table. It
downloads nothing and needs no credentials. Output is written to
`artifacts/evaluation.json`; `artifacts/` is git-ignored.

## Run a real model

Model weights download only when a command gets `--download`. Checkpoints take
roughly 0.4-2.7 GB each.

```sh
uv sync --locked --all-extras
uv run python -m ner_demo.quickstart --model gliner2 --examples --download
HF_HUB_OFFLINE=1 uv run python -m ner_demo.quickstart --model gliner2 --examples
```

`--examples` runs one English, one German, and one Russian sentence and prints
each span with its character offsets and the model's score. They are functional
checks of offsets and repeated mentions, not an accuracy test. Without
`--examples` the quickstart prints the spans for one English sentence.

## Supported models

`Run name` is the value for `--model` (quickstart) and `--models` (evaluation).

| Run name | Family (role) | Checkpoint @ revision | Languages; task | Extra | Command | Status | Limitations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `rules` | Literal dictionary (smoke test) | - | any; NER | - | `uv run python -m ner_demo.evaluate` | validated | Matches four known strings; no general recall. |
| `gliner` | GLiNER v2.1 (urchade) (baseline) | `urchade/gliner_medium-v2.1` @ `40ec419335d09393f298636f471328b722c6da9e` | en; NER | models | `uv run --extra models python -m ner_demo.quickstart --model gliner --examples --download` | validated | Article reproducibility baseline; only model with a validated ONNX FP32/INT8 path. |
| `gliner25` | GLiNER v2.5 (gliner-community) (current) | `gliner-community/gliner_small-v2.5` @ `f227d3cd637bd4e6757ae143935316d062393341` | en; NER | models | `uv run --extra models python -m ner_demo.quickstart --model gliner25 --examples --download` | validated | Model generation v2.5, unrelated to Fastino GLiNER2.5. |
| `gliner-bi` | GLiNER bi-encoder v2.0 (Knowledgator) (current) | `knowledgator/gliner-bi-small-v2.0` @ `3d74c1bf459b8b1c0be1ecbddd679416ce005418` | en; NER | models | `uv run --extra models python -m ner_demo.quickstart --model gliner-bi --examples --download` | validated | Encodes the labels on every call. |
| `gliner-bi-cached` | GLiNER bi-encoder v2.0 (Knowledgator) (current) | `knowledgator/gliner-bi-small-v2.0` @ `3d74c1bf459b8b1c0be1ecbddd679416ce005418` | en; NER | models | `uv run --extra models python -m ner_demo.quickstart --model gliner-bi-cached --examples --download` | validated | Label embeddings computed once per run and reused. |
| `gliner-multi` | GLiNER v2.1 multilingual (urchade) (current) | `urchade/gliner_multi-v2.1` @ `443d26d654e0324125a96bebd8e796c14ff2efe6` | multilingual (card lists no languages; mDeBERTa-v3 backbone); NER | models | `uv run --extra models python -m ner_demo.quickstart --model gliner-multi --examples --download` | validated | No per-language accuracy published for v2.1; the English/German/Russian examples here are functional checks only. |
| `gliner2` | GLiNER2.5 (Fastino) (current) | `fastino/gliner2.5-base-v1` @ `54785d51df8d86a0d8f2eb212fec15871aad2c4d` | en; NER | gliner2 | `uv run --extra gliner2 python -m ner_demo.quickstart --model gliner2 --examples --download` | validated | Separate `gliner2` package. Overlaps are resolved per label, so spans with different labels may overlap. NER only here. |
| `encoder` | BERT token classification (CoNLL-2003) (baseline) | `dslim/bert-base-NER` @ `d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc` | en; NER | models | `uv run --extra models python -m ner_demo.quickstart --model encoder --examples --download` | validated | Fixed PER/ORG/LOC/MISC labels; MISC is dropped. Older supervised baseline, not a current recommendation. |
| `llm` | OpenAI structured outputs (paid, opt-in) (baseline) | `gpt-6-luna` | multilingual (provider claim); NER | teacher | `uv run --extra teacher python -m ner_demo.evaluate --models llm --allow-paid --repeats 1 --warmups 0` | validated | Paid; capped at 32 documents per run. The model writes the character offsets itself, and most did not match their text (docs/results.md). |
| `streaming` | GLiNER Streaming PII (Knowledgator) (current) | `knowledgator/gliner-stream-pii-v1.0` @ `e871777dc4b3b688747a0433fff8d94a36fcc7b0` | en; PII (streaming) | models | `uv run --extra models python -m ner_demo.streaming --download` | validated | Buffering demo only; detector recall is not measured here. |

Status: **validated** means loaded from the pinned revision and run on real text
here on 2026-10-22, offline (`HF_HUB_OFFLINE=1`); **contract** means covered only
by offline tests with fakes or mocks. Checkpoints, revisions, and licenses were
verified on 2026-10-22. Sources, integration details, and the models considered
but not integrated (including Otter and GLiNER-X) are in
[docs/models.md](docs/models.md). Model licenses are those of the checkpoints;
this repository's MIT license covers only its own code.

## Compare models

Import pinned Few-NERD slices (CC-BY-SA-4.0), choose thresholds on the validation
slice, and compare on the test slice:

```sh
uv run python -m ner_demo.data --split validation --limit 500
uv run python -m ner_demo.data --split test --limit 2000 \
  --exclude artifacts/few-nerd-validation.json --output artifacts/few-nerd-test-2000.json
uv run python -m ner_demo.evaluate --dataset artifacts/few-nerd-validation.json \
  --models gliner gliner25 gliner-bi gliner-bi-cached gliner-multi gliner2 \
  --sweep 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.97 --repeats 1 --warmups 0 \
  --output artifacts/thresholds-validation-500.json
uv run python -m ner_demo.evaluate --dataset artifacts/few-nerd-test-2000.json \
  --models rules encoder gliner gliner25 gliner-bi gliner-bi-cached gliner-multi gliner2 \
  --thresholds-from artifacts/thresholds-validation-500.json --repeats 1 \
  --output artifacts/comparison-test-2000.json
```

Add `--download` to a command the first time it needs a checkpoint that is not cached.
The smaller integration check that the article used is the first 32 test
sentences at a fixed threshold of 0.5:

```sh
uv run python -m ner_demo.data --split test --limit 32
uv run python -m ner_demo.evaluate --dataset artifacts/few-nerd-test.json \
  --models rules encoder gliner gliner25 gliner-bi gliner-bi-cached gliner-multi gliner2 \
  --output artifacts/comparison-test-32.json
```

Each run writes one JSON file with the dataset manifest, per-model configuration
(checkpoint, revision, threshold), environment, every raw prediction, and the
metrics, then prints Markdown tables (`python -m ner_demo.report <file>` prints
them again). The runner refuses to choose thresholds on a `test` split or to
reuse them on the documents they were chosen on. Scoring rules, the
input-length policy, and how to read each column are in
[docs/evaluation.md](docs/evaluation.md).

## Results recorded on 2026-10-22

Thresholds were chosen on 500 validation sentences; scores are on 2,000 test
sentences (3,779 gold spans) of Few-NERD's person/organization/location task.
Apple M4, CPU, batch 1, 4 torch threads, one warmup, one repeat.

| Run name | Label space | Threshold | P | R | F1 | F1 95% CI | Load ms | First call ms | p50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `rules` | fixed | - | 0.167 | 0.001 | 0.003 | 0.001-0.005 | 0 | 0 | 0.0 |
| `encoder` | fixed | - | 0.635 | 0.747 | 0.687 | 0.666-0.705 | 1571 | 476 | 16.7 |
| `gliner` | open | 0.7 | 0.613 | 0.665 | 0.638 | 0.618-0.659 | 3692 | 57 | 37.1 |
| `gliner25` | open | 0.8 | 0.656 | 0.740 | 0.695 | 0.677-0.713 | 2425 | 31 | 24.8 |
| `gliner-bi` | open | 0.5 | 0.729 | 0.675 | 0.701 | 0.681-0.719 | 1134 | 20 | 16.3 |
| `gliner-bi-cached` | open | 0.5 | 0.729 | 0.675 | 0.701 | 0.681-0.719 | 1280 | 15 | 11.9 |
| `gliner-multi` | open | 0.8 | 0.641 | 0.636 | 0.639 | 0.618-0.660 | 6270 | 43 | 39.7 |
| `gliner2` | open | 0.95 | 0.687 | 0.649 | 0.668 | 0.651-0.686 | 2939 | 43 | 35.1 |

Every run attempted and scored all 2,000 documents. The bi-encoder, community
GLiNER v2.5, and the fixed-label BERT baseline have overlapping 95% intervals,
so this slice does not separate them; GLiNER v2.1 medium and multilingual v2.1
score lower. The newest checkpoint, GLiNER2.5, does not have the highest F1
here. The paid `gpt-6-luna` baseline, run on the 32-sentence slice only, scored
F1 0.065 because it miscounted the character offsets of most spans. Per-label
scores, threshold sweeps, the 32-sentence check, ONNX FP32/INT8,
label caching, the multilingual examples, and the article's historical numbers
(kept separate from these) are in [docs/results.md](docs/results.md).

## Other workflows

| Workflow | Command | Details |
| --- | --- | --- |
| LLM annotations (paid, needs `--allow-paid` and `OPENAI_API_KEY`) | `uv run --extra teacher python -m ner_demo.teacher --allow-paid` | [docs/workflows.md](docs/workflows.md#llm-annotations-and-training-input-paid-opt-in) |
| Reviewed annotations to GLiNER training files | `uv run --extra models python -m ner_demo.training artifacts/reviewed.jsonl` | [docs/workflows.md](docs/workflows.md#llm-annotations-and-training-input-paid-opt-in) |
| ONNX FP32/INT8 parity check (`gliner` only) | `uv run --extra models --extra onnx python -m ner_demo.onnx_check --dataset artifacts/few-nerd-test.json --output artifacts/onnx-comparison` | [docs/workflows.md](docs/workflows.md#onnx-fp32-and-int8-validated-for-gliner-only) |
| Streaming PII buffering | `uv run --extra models python -m ner_demo.streaming --download` | [docs/workflows.md](docs/workflows.md#streaming-pii-buffering) |

No command makes a paid request unless it is given `--allow-paid`. Never commit
credentials, weights, datasets, or predictions.

## Tests

```sh
uv run python -m unittest discover -s tests -v       # offline contract tests
NER_LOCAL_CHECKS=1 HF_HUB_OFFLINE=1 uv run --all-extras \
  python -m unittest tests.test_real_models -v        # real pinned weights, opt-in
```

The offline tests use fakes, synthetic documents, and a mock HTTP transport
through the real OpenAI SDK; CI runs them with ruff, the package build, and a
smoke run of the built wheel on Python 3.11 and 3.14. The real-model tests load
cached checkpoints and never download or call a paid API.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, required checks, and how to add
an adapter or evaluation data, and [CHANGELOG.md](CHANGELOG.md) for release
notes. The code is MIT-licensed ([LICENSE](LICENSE)). Few-NERD and every model
checkpoint keep their own licenses.
