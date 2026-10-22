# Models: selection, pins, and compatibility

Checkpoints, revisions, licenses, and package versions were verified on
**2026-10-22** against the Hugging Face Hub API (`/api/models/<id>/revision/<sha>`),
model cards, PyPI, and the upstream GitHub release notes linked below. The pinned
data lives in [`ner_demo/registry.py`](../ner_demo/registry.py); the README table is
generated from it with `python -m ner_demo.registry`.

"Latest" below means the newest verified release within a model family, not the
newest upload on the Hub. A checkpoint counts as supported only with the status
the registry gives it:

| Status | Meaning |
| --- | --- |
| validated | Loaded from the pinned revision and run on real text in this repository (smoke documents, the English/German/Russian examples, Few-NERD, and the opt-in tests in `tests/test_real_models.py`). |
| contract | Adapter code covered only by offline contract tests with fakes or mocks. |
| implemented | Adapter code without tests. |
| blocked | Not integrated; the reason is written down. |

## Model versions are not package versions

| Model generation | Python package that runs it |
| --- | --- |
| GLiNER v2.1 and v2.5 checkpoints (`urchade/*`, `gliner-community/*`) | `gliner` 0.2.29 |
| GLiNER bi-encoder v2.0 (`knowledgator/gliner-bi-*-v2.0`) | `gliner` 0.2.29 |
| GLiNER Streaming PII v1.0 | `gliner` 0.2.29 (session API since 0.2.28) |
| Fastino GLiNER2.5 (`fastino/gliner2.5-*`) | `gliner2` 2.1.0 |

Community **GLiNER v2.5** and Fastino **GLiNER2.5** are unrelated architectures. The
first is a span model from the original GLiNER line (`gliner_config.json`,
`span_mode: markerV0`); the second is a boundary model from the GLiNER2 line
(`config.json`, `architecture: boundary`). Neither package loads the other's
checkpoints.

## Integrated checkpoints

### GLiNER v2.1 baseline: `urchade/gliner_medium-v2.1`

- Revision `40ec419335d09393f298636f471328b722c6da9e` (still `main`), Apache-2.0,
  English, DeBERTa-v3-base backbone, 384-word window.
- Kept as the reproducibility baseline for the article's results. It is the only
  checkpoint with a validated ONNX FP32/INT8 path.
- The checkpoint ships no tokenizer and no nested backbone config. The adapter
  stages `microsoft/deberta-v3-base` at `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`
  so nothing resolves to a floating `main`.
- Source: <https://huggingface.co/urchade/gliner_medium-v2.1>

### Community GLiNER v2.5: `gliner-community/gliner_small-v2.5`

- Revision `f227d3cd637bd4e6757ae143935316d062393341` (still `main`; the 2026-04-28
  commit only added fp16/bf16 weight variants), Apache-2.0.
- No newer community or `urchade` uni-encoder release (v2.6, v3, multi-v2.5) exists.
  The card's `multilingual` tag is not backed by a language list or evaluation,
  and the backbone (DeBERTa-v3-small) is English, so the registry lists `en`.
- The checkpoint saved a resized embedding (128,003 rows); the staged backbone
  config keeps that size.
- Source: <https://huggingface.co/gliner-community/gliner_small-v2.5>

### Bi-encoder: `knowledgator/gliner-bi-small-v2.0`

- Revision `3d74c1bf459b8b1c0be1ecbddd679416ce005418`, Apache-2.0, English. Text
  encoder `jhu-clsp/ettin-encoder-68m` (2,048-word window), label encoder
  `sentence-transformers/all-MiniLM-L12-v2`. Paper: [The Million-Label NER
  (arXiv 2602.18487)](https://arxiv.org/abs/2602.18487).
- The checkpoint does not ship the label encoder's tokenizer, and `gliner` would
  load it from the Hub's `main`. The adapter passes a tokenizer pinned at
  `a50ef00143b4d5391434df20ae11632588ac25be`.
- Two run names share the checkpoint. `gliner-bi` encodes the labels on every
  call (`predict_entities`). `gliner-bi-cached` encodes them once per run with
  `encode_labels` and reuses them with `predict_with_embeds`; the one-time cost is
  reported as `label_cache_ms`, separately from model loading and from per-document
  latency. The opt-in tests check that both return the same spans.
- `gliner` 0.2.29 is required for the cache to take effect: in 0.2.28,
  `batch_predict_with_embeds` passed the embeddings under a keyword the PyTorch
  model ignored, so labels were re-encoded anyway
  ([commit c9cccd4](https://github.com/urchade/GLiNER/commit/c9cccd412bb9e7a9ebd95707afd4ec06c3c86f34),
  released in [v0.2.29](https://github.com/urchade/GLiNER/releases/tag/v0.2.29)).
- Source: <https://huggingface.co/knowledgator/gliner-bi-small-v2.0>

### Multilingual GLiNER: `urchade/gliner_multi-v2.1`

- Revision `443d26d654e0324125a96bebd8e796c14ff2efe6`, Apache-2.0, mDeBERTa-v3-base
  backbone pinned at `a0484667b22365f84929a935b5e50a51f71f159d`. Runs offline with
  the same adapter as the other GLiNER span models.
- The card says "multilingual" without a language list. The backbone's
  pretraining data covers German and Russian. The GLiNER paper
  ([arXiv 2311.08526](https://arxiv.org/abs/2311.08526), Table 3) reports German and
  Russian zero-shot results for an earlier multilingual GLiNER, not for v2.1.
- The English, German, and Russian sentences in `example_documents()` are
  functional checks: they confirm character offsets in Cyrillic and umlaut text,
  repeated mentions, and label mapping. Three sentences are not an accuracy result.
- Source: <https://huggingface.co/urchade/gliner_multi-v2.1>

### Fastino GLiNER2.5: `fastino/gliner2.5-base-v1`

- Revision `54785d51df8d86a0d8f2eb212fec15871aad2c4d`, Apache-2.0, English,
  DeBERTa-v3-base boundary model. Fastino's README names it the default English
  model. Package: `gliner2[local]==2.1.0`, the first release that loads GLiNER2.5
  checkpoints with transformers 5
  ([release notes](https://github.com/fastino-ai/GLiNER2/releases/tag/v2.1.0)).
  Paper for the GLiNER2 line: [arXiv 2507.18546](https://arxiv.org/abs/2507.18546).
- The adapter loads the local pinned snapshot through `AutoExtractor`. Given a Hub
  ID, `gliner2` 2.1.0 would load the tokenizer from `main` whatever the revision.
- It always passes `include_spans=True, include_confidence=True`. Without spans,
  `gliner2` deduplicates entities by lower-cased text, so "John met John." would
  return one "John" and no offsets.
- Returned offsets are Python character offsets into the input. `gliner2` appends
  a period to text that does not end in `.`, `!`, or `?`; a span ending on that
  added period would fall outside the input and the scorer would reject it.
- `gliner2` never truncates unless asked to. The adapter rejects texts above the
  checkpoint's 4,096-word `max_len` instead of sending them through one window.
- Overlaps are resolved per label, so spans with different labels may overlap;
  the GLiNER span adapters use `flat_ner=True`, which allows no overlaps at all.
- If a record fails preprocessing, `gliner2` substitutes an empty fallback record
  and logs a warning. With valid labels and non-empty text this path is not
  reached here, but it is upstream behavior to know about.
- Loading prints "You are using a model of type gliner2 to instantiate a model of
  type extractor": the 2026-10-08 card commit renamed `model_type` in
  `config.json`; the weights did not change.
- Source: <https://huggingface.co/fastino/gliner2.5-base-v1>

### Streaming PII: `knowledgator/gliner-stream-pii-v1.0`

- Revision `e871777dc4b3b688747a0433fff8d94a36fcc7b0`, Apache-2.0, English. Still
  the only streaming GLiNER checkpoint on the Hub; later commits edit the card.
  Requires `gliner` 0.2.28 or newer for `inference(..., session_id=...)` and
  `clear_session`.
- Used only by the buffering demo. PII evaluation is kept separate from the
  general-domain comparison, and detector recall is not measured here.
- Source: <https://huggingface.co/knowledgator/gliner-stream-pii-v1.0>

### Token-classification baseline: `dslim/bert-base-NER`

- Revision `d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc` (2024), MIT, English, fixed
  CoNLL-2003 labels. PER, ORG, and LOC map to the shared labels; MISC is dropped.
- An older supervised baseline, not a current recommendation. It is not
  open-vocabulary: the comparison can only use labels it already knows.
- Aggregation is `first`: one tag per word, from its first sub-token, as in the
  BERT paper's NER setup. The earlier `simple` setting split words such as
  "Merkel" into "Merk" + "el" (22 word-splitting spans on the 32 Few-NERD test
  sentences). The setting was chosen for that reason, not tuned on test data.
- Source: <https://huggingface.co/dslim/bert-base-NER>

### Paid LLM baseline: OpenAI `gpt-6-luna`

- `gpt-6-luna` replaces `gpt-5.4-mini`. It is current and not deprecated on
  2026-10-22 ([model page](https://developers.openai.com/api/docs/models/gpt-6-luna),
  [deprecations](https://developers.openai.com/api/docs/deprecations)). OpenAI
  publishes no dated snapshot for it, so the alias can change; every call saves the
  model ID the API returns.
- Chat Completions with a strict `json_schema` response format and
  `max_completion_tokens=2048`. The model's default reasoning effort is `medium`,
  so the request sets `reasoning_effort="none"` explicitly. All three are
  documented for this model.
- Price basis recorded in `ner_demo/teacher.py`: USD 0.10 input, 0.01 cached input,
  0.125 cache write, and 0.50 output per million tokens (standard tier, prompts up
  to 272K tokens).
- SDK: `openai` 3.36.1. Since 3.0 the SDK sends requests through httpx2 instead of
  httpx, so the offline tests mock it with `httpx2.MockTransport`
  ([migration notes](https://github.com/openai/openai-python/blob/main/httpx2.md)).
- A failed call keeps the exception type and the provider's error code, for
  example `RateLimitError: credit_balance_exhausted`, but not the message, which
  can echo request text.
- Not run in this repository: no paid request completed. Every paid path needs
  `--allow-paid`; offline tests send the real SDK request through a mock HTTP
  transport, including a quota error.

## Considered, not integrated

| Candidate | Why not here |
| --- | --- |
| Otter (`whoisjones/otter-cross-mmbert` @ `8729188e4f5fc7948d0e9dfd7d7e6d36c2e7270d`, Apache-2.0; [arXiv 2601.06347](https://arxiv.org/abs/2601.06347)) | Public checkpoints with German and Russian in the training data, but no package: inference needs `trust_remote_code=True`, which executes the repository's model code. Its tokenizer loader silently falls back to the unpinned base tokenizer on any error, and inputs over 1,024 tokens are truncated silently. A candidate for a later, separately reviewed adapter. |
| `knowledgator/gliner-x-*` (multilingual) | Uses a `stanza` word splitter that downloads unpinned language models at inference time; the card lists German but not Russian. |
| `fastino/gliner2.5-multi-v1`, `fastino/gliner2-multi-v1` | Same adapter would work, but the 2.5 card names no languages and the GLiNER2 card claims German, not Russian. Not validated here. |
| `gliner-community/gliner_{medium,large}-v2.5`, `knowledgator/gliner-bi-{edge,base,large}-v2.0`, `fastino/gliner2.5-small-v1` | Other sizes of integrated families. One CPU-friendly size per family keeps the comparison small; adding a size is a registry entry plus a validation run. |
| Knowledgator PII models, `nvidia/gliner-PII`, GLiNER-Relex, GLiNER Decoder, GLiNER-Linker, GLiFormer | Different tasks or packages (PII span sets, relations, label generation, linking). Out of scope for the general-domain span comparison. |

## Dependency notes

- All optional extras resolve together in `uv.lock`, so `uv sync --locked
  --all-extras` installs one consistent environment. `gliner2` is a separate extra
  because it adds `peft` and `accelerate`.
- Pins: `gliner==0.2.29`, `gliner2[local]==2.1.0`, `transformers==5.1.0`,
  `torch==2.11.0`, `onnxruntime==1.24.4`, `onnx==1.21.0`, `openai>=3.36.1,<4`.
  `gliner` moved from 0.2.28, `gliner2` was added, and `openai` moved from 2.30.0
  to 3.x; transformers stays at 5.1.0, inside the range both GLiNER packages
  accept.
- On Linux x86_64 the locked `torch==2.11.0` wheel from PyPI depends on CUDA
  libraries (`nvidia-*` packages, several GB), even for CPU-only use. Pointing
  torch at PyTorch's CPU index would change the lock for every user; that choice
  is left to the maintainer.
- transformers 5.1.0 warns that the DeBERTa-v3 tokenizer has "an incorrect regex
  pattern". The check flags any Hub tokenizer saved by transformers 4.57.2 or
  older; it applies no change unless `fix_mistral_regex=True` is passed. GLiNER
  results reproduce the article's numbers exactly with the warning present.
- `gliner`'s word splitter treats a combining mark (decomposed Unicode, NFD) as a
  separate token, so `"Zoë"` can come back as `"Zoe"`. Normalize text to NFC
  before annotation and inference. `gliner2` keeps the mark.
- Python 3.11-3.14 is supported; the 0.3.0 entry in
  [CHANGELOG.md](../CHANGELOG.md) lists what was run on each version. `gliner2`
  2.1.0 declares Python classifiers only up to 3.12, but its real-model tests
  passed on 3.13 and 3.14 as well.
