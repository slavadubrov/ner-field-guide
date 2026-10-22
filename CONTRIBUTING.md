# Contributing

This is a small reference project. Changes that make results more trustworthy or
easier to reproduce are more useful than new features.

## Development setup

Python 3.11-3.14 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --locked --extra teacher --extra onnx   # offline tests; no model weights
uv sync --locked --all-extras                   # everything, including torch
```

## Required checks

CI runs these on Python 3.11 and 3.14; run them before opening a pull request:

```sh
uv run ruff check .
uv run ruff format --check .
uv run python -m unittest discover -s tests -v
uv run python -m scripts.run_all
uv build
```

The unit tests are offline contract tests: fakes, synthetic documents, and a mock
HTTP transport for the OpenAI SDK. They never download weights or make paid calls.
Real-model checks are opt-in and need the pinned checkpoints in the local cache:

```sh
NER_LOCAL_CHECKS=1 HF_HUB_OFFLINE=1 uv run --all-extras \
  python -m unittest tests.test_real_models -v
```

Run them when you change an adapter, a pin, or a dependency, and say in the pull
request that you did and on which hardware.

## Adding a model adapter

1. Add an entry to `MODELS` in `ner_demo/registry.py` with the exact checkpoint,
   a 40-character revision (not `main`), license, languages, task, the extra that
   installs its package, sources, and `verified` date. Pin every companion file the
   checkpoint loads from another repository (tokenizer, backbone config).
2. Reuse an existing adapter when the checkpoint fits one. Otherwise add a branch
   to `make_runner` in `ner_demo/models.py` that returns spans in the shared
   contract: character offsets into the input, the requested label, the covered
   text, and the model's own `score`. Do not repair or guess offsets, do not drop
   predictions that fail to align, and reject inputs the model would truncate.
3. Add an offline contract test for any output conversion (see
   `Adapters.test_gliner2_keeps_repeats_and_native_scores`) and add the model to
   the relevant real-model tests.
4. Set `status` to what you actually ran: `validated` only after loading the
   pinned weights offline and running the real-model tests and an evaluation;
   `contract` if only offline tests cover it.
5. Regenerate the README table with `uv run python -m ner_demo.registry` and paste
   the output over the old table. A unit test fails while they differ.
6. Do not copy scores from model cards into the README. Report only what an
   evaluation command produced, with its output file.

## Adding evaluation data

`--dataset` accepts any JSON file shaped like the Few-NERD import:

```json
{"manifest": {"labels": ["person"], "split": "test", "documents_sha256": "..."},
 "documents": [{"document_id": "doc-1", "text": "John met John.",
                "entities": [{"start": 0, "end": 4, "text": "John", "label": "person"}]}]}
```

`documents_sha256` is `ner_demo.core.digest(documents)`. Record the source,
revision, license, and selection rule in the manifest, keep development and test
documents in separate files, and set `split` so the runner can refuse threshold
selection on test data. Do not commit datasets, weights, or predictions;
`artifacts/` is ignored.

## Reporting a reproducible bug

Open an issue with:

- the exact command and the full error or the unexpected output;
- `manifest.packages`, `manifest.git_commit`, `manifest.git_dirty`, `cpu`, and
  `device` from the output JSON, or `uv pip list` if no output was written;
- the operating system, Python version, and whether `HF_HUB_OFFLINE` was set;
- for a wrong span, the document text (or a minimal text that shows it), the
  returned entity, and the expected offsets.

Never paste API keys or private documents.

## Useful contributions

- Real-model validation on Linux or a GPU, with the output JSON attached.
- Development and test sets from other domains or languages, with licenses that
  allow redistribution of derived data.
- An adapter for a model listed under "Considered, not integrated" in
  `docs/models.md`, with the blocker there resolved.
- A tested windowing policy for documents longer than a model's window.
- Fixes to anything in `docs/` that no longer matches the code.

## License

By contributing you agree that your contribution is licensed under the MIT
License in `LICENSE`.
