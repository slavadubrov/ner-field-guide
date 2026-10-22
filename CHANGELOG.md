# Changelog

## 0.3.0 - unreleased

First tagged release. Earlier work is in the git history. `pyproject.toml` on
`main` said 0.2.0, but 0.2.0 was never tagged or published, so this release
starts at 0.3.0.

### Added

- A pinned model registry, `ner_demo/registry.py`: exact checkpoint, 40-character
  revision, pinned companion files, family, adapter, languages, task, license,
  install extra, sources, verification date (2026-10-22), and validation status.
  The README model table is generated from it, and a test fails when the two
  differ.
- Adapters for current checkpoints, each validated with real weights:
  - `gliner-bi` and `gliner-bi-cached`: Knowledgator GLiNER bi-encoder v2.0
    (`knowledgator/gliner-bi-small-v2.0`), with labels encoded per call or once
    per run; the one-time label encoding is reported as `label_cache_ms`.
  - `gliner-multi`: multilingual GLiNER v2.1 (`urchade/gliner_multi-v2.1`).
  - `gliner2`: Fastino GLiNER2.5 (`fastino/gliner2.5-base-v1`) through the
    `gliner2` package, in a new `gliner2` extra.
- `ner_demo.data --exclude`: skips rows whose text occurs in another imported
  file. Few-NERD repeats at least one sentence across its validation and test
  splits; the new threshold guard found it.
- English, German, and Russian functional examples:
  `python -m ner_demo.quickstart --model <name> --examples`.
- Threshold selection on development data: `evaluate --sweep` and
  `--thresholds-from`. Only a threshold whose run scored every development
  document can be selected. The runner refuses sweeps on a test split, reuse on
  the documents the thresholds were chosen on, and thresholded models without a
  development threshold; the report flags selections at the edge of the grid.
- Attempted, scored, failed, and missing document counts (a model that fails to
  load counts its documents as missing and is marked "load failed"); a 95%
  document-bootstrap interval for micro F1; first-call latency;
  label-cache time; a flag when repeats disagree; CPU model, thread count, and
  uncommitted-change flag in every output; `python -m ner_demo.report`.
- Opt-in real-model tests (`tests/test_real_models.py`), MIT `LICENSE`,
  `CONTRIBUTING.md`, and `docs/` (models, evaluation method, workflows).
- Package metadata: readme, SPDX license, author, project URLs, classifiers.

### Changed

- `gliner` 0.2.28 -> 0.2.29. In 0.2.28 precomputed bi-encoder label embeddings
  were ignored, and `from_pretrained` did not accept a pinned label tokenizer.
- The token-classification baseline aggregates with `first` instead of `simple`.
  `simple` split words into partial entities ("Merk" + "el"); on the 32 Few-NERD
  test sentences word-splitting spans fell from 22 to 1 and micro F1 rose from
  0.567 to 0.677. The article's baseline numbers used `simple`.
- Every GLiNER checkpoint loads from its pinned local snapshot only
  (`local_files_only=True`); the bi-encoder's label tokenizer is pinned too.
- The paid LLM baseline moves from `gpt-5.4-mini` to `gpt-6-luna` (price basis
  checked on 2026-10-22; OpenAI publishes no dated snapshot for it) and to the
  `openai` 3.x SDK, which sends requests through httpx2; the offline tests now mock
  httpx2. A failed call records the provider's error code with the exception
  type, so an exhausted quota is no longer reported as a plain rate limit.
- Evaluation output: per-model configuration with its own threshold (`null` for
  models without one) replaces the single global threshold.
- `ner_demo.data` writes `artifacts/few-nerd-<split>.json` by default (it always
  wrote `few-nerd-test.json`, so importing another split overwrote the test file)
  and records the split size.
- CI runs on Python 3.11 and 3.14, runs the ONNX graph tests (previously skipped
  because the `onnx` extra was not installed), and checks that the wheel ships
  the license. It runs on pull requests and on pushes to `main`, so a pull
  request commit is checked once instead of twice.
- The sdist is built from an explicit file list.

### Fixed

- Streaming: a whitespace-only chunk desynchronized the model session from the
  buffer (the session API skips such chunks), so the next span failed offset
  validation and the session aborted. Whitespace is now held and sent with the
  next chunk that has visible text.
- Streaming: a session under the 16,000-character cap but over the model's
  8,192-word limit passed the token check, and the final full-text pass silently
  kept only the first 8,192 words, so the tail was released without detection.
  Such sessions are now rejected before release.
- Evaluation failed a whole LLM document when one span had bad offsets, so the
  document's correct spans counted as false positives. Such spans are now scored
  one by one, as for every other adapter; refusals, truncated output, and API
  errors still fail the document.
- Provenance recorded the git commit of whatever repository the caller was in
  (for an installed copy, often the user's own project), and the run crashed when
  `git` was not installed. The commit is now recorded only when this source file
  is tracked by the repository; otherwise it is `null`.
- ONNX package validation imported transformers before its file checks.
- The sdist included untracked local files (hatchling ignores
  `.git/info/exclude`).

### Validation

Run on 2026-10-22 on an Apple M4 (10 logical CPUs, macOS 27.2), CPU only.

- Offline checks (no weights, credentials, or paid calls): ruff lint and format,
  the unit tests, `scripts.run_all`, the package build, and the offline smoke
  from the built wheel in a clean virtual environment, on Python 3.11, 3.12,
  3.13, and 3.14. CI's pinned uv 0.11.26 accepts the lock file.
- Real weights, loaded offline from the pinned revisions: the six tests in
  `tests/test_real_models.py` passed on Python 3.11, 3.12, 3.13, and 3.14
  (offsets in Latin, German, and Cyrillic text, repeated mentions, native
  scores, cached and uncached bi-encoder agreement, rejection of over-long input,
  GLiNER's training collator, and streaming buffering with the real checkpoint).
- Few-NERD, three coarse types: thresholds chosen on 500 validation sentences,
  then all eight general-domain runs attempted and scored all 2,000 test
  sentences with no failures. Micro F1 with 95% document-bootstrap intervals:
  bi-encoder 0.701 (0.681-0.719), community GLiNER v2.5 0.695 (0.677-0.713), BERT
  baseline 0.687 (0.666-0.705), GLiNER2.5 0.668 (0.651-0.686), multilingual
  GLiNER v2.1 0.639 (0.618-0.660), GLiNER v2.1 medium 0.638 (0.618-0.659).
- The article's 32-sentence counts for `gliner` (39/38/15) and `gliner25`
  (44/35/10) reproduce exactly with `gliner` 0.2.29.
- ONNX: FP32 returned the PyTorch spans exactly; INT8 changed 4 of 32 sentences
  and added 3 false positives at equal recall; package sizes match the article's
  run byte for byte.
- Bi-encoder label cache: p50 11.9 ms cached vs 16.3 ms per-call on the 2,000
  sentences, identical predictions; on `gliner` 0.2.28 the two paths took the
  same time.

- Every README command was then run verbatim in a fresh clone of the release
  commit `a85ffa3`: offline example, cached quickstart, data import, sweep, both
  comparisons, ONNX check, streaming demo, and both test suites. Counts,
  selected thresholds, and intervals matched the earlier run exactly.

- Paid baseline, `gpt-6-luna`, 32 test sentences (33 requests with the teacher
  check, about USD 0.0014): every request finished, but exact-span F1 was 0.065
  (0.013-0.141) because 65 of the 70 returned spans had character offsets that did
  not match their text.

Full tables: [docs/results.md](docs/results.md).

### Known limitations

- All measurements come from one machine (Apple M4, CPU, batch 1) and one
  English Wikipedia dataset restricted to three entity types. The intervals cover
  document resampling only, not other data, labels, or hardware.
- Not run: GPU and MPS devices, and Linux or Windows hosts beyond CI's offline
  checks.
- The LLM baseline asks the model for character offsets, and `gpt-6-luna`
  miscounted most of them, so its exact-span score says more about offset
  arithmetic than about entity recognition. Deriving offsets in code from a
  returned quotation would avoid this; it is not implemented.
- Inputs longer than a model's window are rejected; there is no windowing.
- German and Russian support is checked with three functional sentences only; no
  multilingual accuracy was measured. Otter and GLiNER-X are not integrated; see
  [docs/models.md](docs/models.md).
- GLiNER2.5 resolves overlaps per label, while the GLiNER span adapters allow no
  overlaps at all, so decoding differs between families.
- On Linux x86_64 the locked torch wheel installs CUDA libraries even for CPU
  use. A CPU-only index would change the lock and is left to the maintainer.
- `gliner`'s word splitter can leave a combining mark (NFD text) outside a span;
  normalize text to NFC before annotation and inference.
- The streaming demo buffers the whole session, so release delay grows with the
  input, and it does not measure PII recall.

### Reproduce

```sh
uv sync --locked
uv run python -m unittest discover -s tests -v
uv run python -m scripts.run_all
uv sync --locked --all-extras
uv run python -m ner_demo.quickstart --model gliner2 --examples --download
```

The model comparison commands are in the README.
