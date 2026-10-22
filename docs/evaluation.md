# Evaluation method

This page describes what `python -m ner_demo.evaluate` measures, the rules it
enforces, and how to read its output. The scorer is `ner_demo.core.score`; the
runner is `ner_demo.evaluate`.

## Span contract and scoring rules

Each document has a unique `document_id`, `text`, and `entities`. An entity is:

```json
{"start": 0, "end": 4, "text": "John", "label": "person", "score": 0.99}
```

- Offsets are Python Unicode character indices, half-open `[start, end)`, and
  `text[start:end]` must equal the entity text. `score` is optional and holds the
  model's own confidence when it has one; scores are not calibrated across models.
- A match needs the same document, start, end, and label. Repeated mentions count
  separately. The scorer accepts nested spans; the comparison asks models for
  flat output where they support it (see decoding policies below).
- An unmatched prediction is a false positive; an unmatched gold span is a false
  negative. Duplicate predictions add false positives. Duplicate gold spans,
  duplicate result IDs, and result IDs that are not in the dataset reject the run.
- A prediction with an unknown label or invalid offsets counts as a false
  positive and marks its document as not scored. Adapters never repair offsets,
  and the scorer never searches the text for a matching substring.
- Missing, failed, and skipped documents stay in the gold denominator. The
  `headline` metrics are withheld (`null`) unless every document was scored.
- Micro metrics sum counts over labels. Macro metrics average over all requested
  labels, including labels with no gold spans.

Every run reports four document counts next to the metrics: **attempted** (the
adapter was called), **scored** (completed with only valid spans), **failed**
(the adapter raised or returned an error), and **missing** (never attempted). A
model that fails to load is never called, so all its documents count as missing
and the report marks the row "load failed" with the exception type.

## Dataset: pinned Few-NERD slices

`python -m ner_demo.data` downloads one Parquet shard of
[Few-NERD](https://huggingface.co/datasets/DFKI-SLT/few-nerd) (`supervised`
config, revision `205f3e9c9f3577ea2561d43f2f62dc249ab92d5b`, CC-BY-SA-4.0) with a
100 MB cap, keeps the first N rows of the chosen split, and writes
`artifacts/few-nerd-<split>.json` with a manifest (source URL and SHA-256, label
mapping, row counts, and a hash of the converted documents). It runs no remote
dataset code.

- Text is the tokens joined by one ASCII space, so offsets refer to that
  sentence, not to the Wikipedia page.
- Only the coarse person, organization, and location categories are kept; other
  categories are dropped, never relabeled. This is a restricted three-label task,
  not full Few-NERD scoring. A model asked for "location" can still predict a
  building or an event location that Few-NERD files under another category, and
  that counts as a false positive.
- Few-NERD uses IO tags, so two adjacent entities with the same fine type are
  merged into one gold span.
- The split rows are already shuffled across articles, so the first N rows are an
  unclustered slice. Keep splits in separate files: thresholds are chosen on
  `validation`, and the final comparison runs on `test`.
- Few-NERD repeats some sentences across splits: validation row 414 and test row
  207 are the same sentence with the same labels. Import the test slice with
  `--exclude artifacts/few-nerd-validation.json`, which skips rows whose text is
  in that file and keeps taking rows until N are selected; the excluded file must
  match its own manifest hash, and the new manifest records that hash and the
  number of skipped rows. The evaluation runner
  found this duplicate: it refuses thresholds chosen on overlapping texts.
  Exact duplicates within one split (two sentences in the first 2,000 test rows)
  stay, and each copy is scored as its own document.

## Threshold selection

GLiNER-family adapters (`gliner`, `gliner25`, `gliner-bi`, `gliner-bi-cached`,
`gliner-multi`, `gliner2`) keep spans at or above a score threshold. Fixed-label
adapters (`encoder`, `rules`) and the LLM have none, and their threshold is
recorded as `null`.

```sh
# Development data: score each threshold, keep the best micro F1 per model.
python -m ner_demo.evaluate --dataset artifacts/few-nerd-validation.json \
  --models gliner gliner25 --sweep 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.97 \
  --repeats 1 --warmups 0 --output artifacts/thresholds.json
# Test data: reuse those thresholds.
python -m ner_demo.evaluate --dataset artifacts/few-nerd-test.json \
  --models encoder gliner gliner25 --thresholds-from artifacts/thresholds.json
```

The runner refuses a sweep on a dataset whose manifest says `split: test`, refuses
`--thresholds-from` when any document text overlaps the documents the thresholds
were chosen on, and refuses a thresholded model that has no development
threshold. Only a threshold whose run scored every development document can be
selected; a model that failed to load or rejected a document exports no
threshold, so the test run refuses it instead of using an unmeasured value. Ties
go to the first listed threshold. The test output records the SHA-256 of the
sweep file and the dataset manifest it was chosen on.

The report marks a selection at either end of the grid ("grid edge: widen the
sweep"), because the best threshold may lie outside it. This happened here: on a
0.3-0.7 grid, four of six models selected 0.7, and on 0.3-0.9 GLiNER2.5 still
selected 0.9. The documented grid, 0.3-0.9 plus 0.95 and 0.97, brackets every
model's best value on the 500 validation sentences. Scores are not calibrated
across models, so one shared threshold would favor some models over others.

## What "comparable" means here

One run evaluates every selected model on the same documents, labels, scoring
rules, input-length policy, and split. Differences that remain are properties of
the models and are recorded rather than hidden:

| Adapter | Label space | Decoding | Input-length policy (rejects; never truncates) |
| --- | --- | --- | --- |
| `gliner`, `gliner25`, `gliner-multi` | open (label names in the prompt) | flat across labels (`flat_ner=True`) | over `max_len` words (384, or 768 for `gliner25`), or over 512 subwords including the label prompt |
| `gliner-bi`, `gliner-bi-cached` | open (separate label encoder) | flat across labels | over 2,048 words, over 7,999 text subwords, or a label over 512 label-encoder tokens |
| `gliner2` | open | flat within each label; different labels may overlap | over 4,096 words |
| `encoder` | fixed CoNLL labels; MISC dropped | BIO with `first` aggregation | over 512 subwords |
| `llm` | open | strict JSON schema with the quoted text and its appearance number; offsets computed in code; a quote that does not resolve counts as a false positive; refusals, truncated output, and API errors fail the document | the CLI caps documents at 4,000 characters |

Open-vocabulary models are prompted with the label names `person`,
`organization`, and `location`. These are common types in their training data,
so the comparison is not a zero-shot test of unseen types. The fixed-label
encoder was trained for exactly these types on news text. The shared three-label
subset is the only task all models can be scored on.

## Uncertainty

The report gives a 95% interval for micro F1: a percentile bootstrap over
documents (1,000 resamples with replacement, fixed seed), using the same
resamples for every model in a run; a resample with no gold and no predicted
spans has no defined F1 and is skipped. When two intervals overlap, the slice cannot
separate those models; that does not show they are equal. Resampling documents
keeps the spans of one sentence together. No paired significance test is
implemented, and the interval does not cover other datasets, label sets, or
thresholds.

## Latency

- Device is CPU unless `--device` says otherwise. Batch size is one document.
- `Load ms` is model loading (`initialization_ms`), including the bi-encoder label
  cache when used. `Label cache ms` is that one-time label encoding on its own.
- `First call ms` is the first warmup call on the first document (cold start). It
  is excluded from p50/p95.
- `p50`/`p95` are over every successful call in every repeat. Failed calls are
  excluded from latency but stay in the metrics as failures.
- The manifest records the CPU model, logical CPU count, torch thread count,
  device, warmups, repeats, Python and package versions, the git commit, and
  whether the working tree had uncommitted changes.

Latency on another machine, thread count, or batch size will differ. Compare
latencies only within one output file.

## Output

`--output` (default `artifacts/evaluation.json`) holds:

- `manifest`: dataset manifest, document IDs and hashes, labels, per-model
  configuration (checkpoint, revision, pinned companions, threshold, label
  cache), threshold selection, device and repetition settings, and environment.
- `results.<model>`: load time, label-cache time, warmup times and errors, every
  repeat's raw predictions per document (status, entities, latency, error type),
  per-repeat metrics, a flag when metrics differ between repeats, and latency
  percentiles. Sweeps add `sweep` (micro metrics per threshold).
- `selected_thresholds` for `--sweep` runs.

The runner prints Markdown tables at the end; `python -m ner_demo.report
<output.json>` prints them again. Error rows keep only the exception type, not
the message, because provider errors can contain request text.

## Configurations

| Configuration | Documents | Thresholds | Repeats | Purpose |
| --- | --- | --- | --- | --- |
| Smoke | 3 synthetic documents (no `--dataset`) | fixed 0.5 | 3 | Offline regression check; not a model ranking |
| Small | first 32 Few-NERD test sentences | fixed 0.5 | 3 | Integration check; matches the article's runs |
| Larger | first 500 validation sentences for thresholds, first 2,000 test sentences for the comparison | development sweep 0.3-0.97 | 1 | The documented comparison |

The exact commands are in the README. The larger configuration is still a
three-label slice of one English Wikipedia dataset scored on one machine. Treat
its results as a check of these adapters under these rules, not as a general
ranking of the model families.
