# Results recorded in this repository

Every number on this page was produced on **2026-10-22** by the commands in the
README, on one machine: Apple M4 (10 logical CPUs; torch used 4 threads), macOS
27.2, CPU only, batch size 1, Python 3.13.3, `gliner` 0.2.29, `gliner2` 2.1.0,
transformers 5.1.0, torch 2.11.0, ONNX Runtime 1.24.4. Models loaded offline
(`HF_HUB_OFFLINE=1`) from the pinned revisions. The local-model tables were
produced by running the README commands in a fresh clone of commit `a85ffa3` (the
outputs record `git_dirty: false`); an earlier run from the uncommitted working
tree gave identical counts, thresholds, and intervals. The paid LLM table was
produced afterwards from a clean checkout of commit `5b3c7da` with `openai`
3.36.1.

The task is Few-NERD's three coarse types (person, organization, location) on
English Wikipedia sentences. These results check the adapters under one set of
rules; they are not a general ranking of the model families. No model-card
scores are copied here.

## Larger configuration

### Thresholds chosen on 500 validation sentences

Data: DFKI-SLT/few-nerd@205f3e9 split=validation; 500 documents; labels: person, organization, location
Run: Apple M4, 10 logical CPUs, torch threads 4, device cpu, batch 1, 0 warmup, 1 repeats; git a85ffa3
Thresholds: development sweep: 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97

| Model | Checkpoint | Labels | Threshold | P | R | F1 | F1 95% CI | TP/FP/FN | Attempted | Scored | Failed | Missing | Load ms | Label cache ms | First call ms | p50 ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gliner | urchade/gliner_medium-v2.1@40ec419 | open | 0.70 | 0.638 | 0.691 | 0.663 | 0.629-0.694 | 660/375/295 | 500 | 500 | 0 | 0 | 3443 | - | - | 37.1 | 49.4 |
| gliner25 | gliner-community/gliner_small-v2.5@f227d3c | open | 0.80 | 0.690 | 0.762 | 0.724 | 0.695-0.753 | 728/327/227 | 500 | 500 | 0 | 0 | 2357 | - | - | 25.1 | 34.6 |
| gliner-bi | knowledgator/gliner-bi-small-v2.0@3d74c1b | open | 0.50 | 0.746 | 0.725 | 0.735 | 0.703-0.766 | 692/235/263 | 500 | 500 | 0 | 0 | 1124 | - | - | 16.9 | 23.2 |
| gliner-bi-cached | knowledgator/gliner-bi-small-v2.0@3d74c1b | open | 0.50 | 0.746 | 0.725 | 0.735 | 0.703-0.766 | 692/235/263 | 500 | 500 | 0 | 0 | 1229 | 122 | - | 12.3 | 19.2 |
| gliner-multi | urchade/gliner_multi-v2.1@443d26d | open | 0.80 | 0.668 | 0.677 | 0.673 | 0.639-0.703 | 647/321/308 | 500 | 500 | 0 | 0 | 5860 | - | - | 39.8 | 51.9 |
| gliner2 | fastino/gliner2.5-base-v1@54785d5 | open | 0.95 | 0.699 | 0.685 | 0.692 | 0.661-0.721 | 654/282/301 | 500 | 500 | 0 | 0 | 2435 | - | - | 35.4 | 44.8 |

| Per-label F1 | person | organization | location |
| --- | --- | --- | --- |
| gliner | 0.756 | 0.537 | 0.677 |
| gliner25 | 0.840 | 0.608 | 0.718 |
| gliner-bi | 0.837 | 0.651 | 0.705 |
| gliner-bi-cached | 0.837 | 0.651 | 0.705 |
| gliner-multi | 0.751 | 0.567 | 0.685 |
| gliner2 | 0.806 | 0.572 | 0.686 |

| Sweep (micro F1) | threshold: F1 | Selected |
| --- | --- | --- |
| gliner | 0.3: 0.614, 0.4: 0.632, 0.5: 0.645, 0.6: 0.654, 0.7: 0.663, 0.8: 0.658, 0.9: 0.591, 0.95: 0.430, 0.97: 0.246 | 0.7 |
| gliner25 | 0.3: 0.669, 0.4: 0.684, 0.5: 0.697, 0.6: 0.707, 0.7: 0.719, 0.8: 0.724, 0.9: 0.722, 0.95: 0.695, 0.97: 0.668 | 0.8 |
| gliner-bi | 0.3: 0.688, 0.4: 0.727, 0.5: 0.735, 0.6: 0.597, 0.7: 0.155, 0.8: 0.000, 0.9: 0.000, 0.95: 0.000, 0.97: 0.000 | 0.5 |
| gliner-bi-cached | 0.3: 0.688, 0.4: 0.727, 0.5: 0.735, 0.6: 0.597, 0.7: 0.155, 0.8: 0.000, 0.9: 0.000, 0.95: 0.000, 0.97: 0.000 | 0.5 |
| gliner-multi | 0.3: 0.603, 0.4: 0.622, 0.5: 0.636, 0.6: 0.655, 0.7: 0.666, 0.8: 0.673, 0.9: 0.631, 0.95: 0.504, 0.97: 0.353 | 0.8 |
| gliner2 | 0.3: 0.601, 0.4: 0.608, 0.5: 0.619, 0.6: 0.628, 0.7: 0.641, 0.8: 0.656, 0.9: 0.688, 0.95: 0.692, 0.97: 0.671 | 0.95 |

Scored = completed with only valid spans. Failed and missing documents stay in the denominator. F1 95% CI = document bootstrap, 1,000 resamples, the same resamples for every model. First call = warmup on the first document.

Every selected threshold lies inside the final grid. On narrower grids four
models first selected the top value; see
[evaluation.md](evaluation.md#threshold-selection).

### Comparison on 2,000 test sentences

3,779 gold spans. One test sentence that also occurs in the validation slice was
skipped with `--exclude`, so the slice is test rows 0-2000 without row 207.

Data: DFKI-SLT/few-nerd@205f3e9 split=test; 2000 documents; labels: person, organization, location
Run: Apple M4, 10 logical CPUs, torch threads 4, device cpu, batch 1, 1 warmup, 1 repeats; git a85ffa3
Thresholds: from artifacts/thresholds-validation-500.json, chosen on DFKI-SLT/few-nerd split=validation

| Model | Checkpoint | Labels | Threshold | P | R | F1 | F1 95% CI | TP/FP/FN | Attempted | Scored | Failed | Missing | Load ms | Label cache ms | First call ms | p50 ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rules | - | fixed | - | 0.167 | 0.001 | 0.003 | 0.001-0.005 | 5/25/3774 | 2000 | 2000 | 0 | 0 | 0 | - | 0.1 | 0.0 | 0.0 |
| encoder | dslim/bert-base-NER@d1a3e8f | fixed | - | 0.635 | 0.747 | 0.687 | 0.666-0.705 | 2823/1621/956 | 2000 | 2000 | 0 | 0 | 1571 | - | 475.9 | 16.7 | 22.0 |
| gliner | urchade/gliner_medium-v2.1@40ec419 | open | 0.70 | 0.613 | 0.665 | 0.638 | 0.618-0.659 | 2513/1584/1266 | 2000 | 2000 | 0 | 0 | 3692 | - | 57.2 | 37.1 | 47.9 |
| gliner25 | gliner-community/gliner_small-v2.5@f227d3c | open | 0.80 | 0.656 | 0.740 | 0.695 | 0.677-0.713 | 2795/1464/984 | 2000 | 2000 | 0 | 0 | 2425 | - | 30.6 | 24.8 | 33.0 |
| gliner-bi | knowledgator/gliner-bi-small-v2.0@3d74c1b | open | 0.50 | 0.729 | 0.675 | 0.701 | 0.681-0.719 | 2551/949/1228 | 2000 | 2000 | 0 | 0 | 1134 | - | 20.3 | 16.3 | 21.9 |
| gliner-bi-cached | knowledgator/gliner-bi-small-v2.0@3d74c1b | open | 0.50 | 0.729 | 0.675 | 0.701 | 0.681-0.719 | 2551/949/1228 | 2000 | 2000 | 0 | 0 | 1280 | 121 | 15.4 | 11.9 | 17.1 |
| gliner-multi | urchade/gliner_multi-v2.1@443d26d | open | 0.80 | 0.641 | 0.636 | 0.639 | 0.618-0.660 | 2403/1344/1376 | 2000 | 2000 | 0 | 0 | 6270 | - | 43.4 | 39.7 | 50.8 |
| gliner2 | fastino/gliner2.5-base-v1@54785d5 | open | 0.95 | 0.687 | 0.649 | 0.668 | 0.651-0.686 | 2454/1117/1325 | 2000 | 2000 | 0 | 0 | 2939 | - | 42.8 | 35.1 | 44.5 |

| Per-label F1 | person | organization | location |
| --- | --- | --- | --- |
| rules | 0.000 | 0.000 | 0.006 |
| encoder | 0.807 | 0.559 | 0.698 |
| gliner | 0.770 | 0.530 | 0.627 |
| gliner25 | 0.837 | 0.595 | 0.679 |
| gliner-bi | 0.844 | 0.622 | 0.655 |
| gliner-bi-cached | 0.844 | 0.622 | 0.655 |
| gliner-multi | 0.761 | 0.550 | 0.615 |
| gliner2 | 0.798 | 0.527 | 0.681 |

Scored = completed with only valid spans. Failed and missing documents stay in the denominator. F1 95% CI = document bootstrap, 1,000 resamples, the same resamples for every model. First call = warmup on the first document.

What these numbers support:

- All eight runs attempted and scored all 2,000 documents. None failed, none was
  missing, and no sentence hit a length limit.
- The intervals of the bi-encoder (0.681-0.719), community GLiNER v2.5 small
  (0.677-0.713), and the BERT baseline (0.666-0.705) overlap: this slice does not
  separate them. GLiNER2.5 base (0.651-0.686) overlaps the lower part of that
  group. GLiNER v2.1 medium and multilingual v2.1 (upper bounds 0.659 and 0.660)
  are below the lower bounds of the bi-encoder and community v2.5.
- The BERT baseline is a fixed-label model trained on CoNLL-2003 news for these
  types (plus MISC). The open-vocabulary models are prompted with the type
  names. This is a shared three-label task, not a zero-shot test.
- The newest checkpoint here, Fastino GLiNER2.5 (August 2026), does not have the
  highest F1 on this slice. One English Wikipedia slice is not enough to
  generalize that in either direction.
- Best thresholds range from 0.5 (bi-encoder) to 0.95 (GLiNER2.5). Scores are not
  calibrated across models, so one shared threshold would favor some of them.
- Latency is per sentence on short inputs (at most 488 characters). Longer
  documents, other hardware, other thread counts, or batching change it.

## Bi-encoder label cache

Same checkpoint, same predictions, three labels:

| Measurement | Labels encoded per call | Labels cached |
| --- | --- | --- |
| 2,000-sentence run, p50 / p95 | 16.3 / 21.9 ms | 11.9 / 17.1 ms |
| One-time label encoding (`label_cache_ms`) | - | 120 ms |
| Interleaved probe, 32 sentences x 5 repeats, `gliner` 0.2.28 | 16.1 ms | 16.1 ms |
| Interleaved probe, 32 sentences x 5 repeats, `gliner` 0.2.29 | 16.0 ms | 11.6 ms |

On `gliner` 0.2.28 the cached path re-encoded the labels, so it saved nothing;
0.2.29 fixed this. With three labels the saving is about 4 ms per sentence. The
large-label-set gains reported in the bi-encoder paper were not measured here.
The probe was a one-off script: it loaded the checkpoint once per `gliner`
version and alternated the two calls on each of the 32 test sentences, five
times, so load changes on the machine affected both paths alike.

## Small configuration: 32 test sentences, threshold 0.5

The article's integration check: same 32 sentences (documents hash
`60666d86...`), fixed threshold, three repeats.

Data: DFKI-SLT/few-nerd@205f3e9 split=test; 32 documents; labels: person, organization, location
Run: Apple M4, 10 logical CPUs, torch threads 4, device cpu, batch 1, 1 warmup, 3 repeats; git a85ffa3
Thresholds: fixed 0.5

| Model | Checkpoint | Labels | Threshold | P | R | F1 | F1 95% CI | TP/FP/FN | Attempted | Scored | Failed | Missing | Load ms | Label cache ms | First call ms | p50 ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rules | - | fixed | - | 0.000 | 0.000 | 0.000 | 0.000-0.000 | 0/0/54 | 32 | 32 | 0 | 0 | 0 | - | 0.1 | 0.0 | 0.0 |
| encoder | dslim/bert-base-NER@d1a3e8f | fixed | - | 0.600 | 0.778 | 0.677 | 0.551-0.780 | 42/28/12 | 32 | 32 | 0 | 0 | 1321 | - | 37.1 | 16.3 | 20.9 |
| gliner | urchade/gliner_medium-v2.1@40ec419 | open | 0.50 | 0.506 | 0.722 | 0.595 | 0.439-0.722 | 39/38/15 | 32 | 32 | 0 | 0 | 3654 | - | 57.5 | 36.9 | 48.3 |
| gliner25 | gliner-community/gliner_small-v2.5@f227d3c | open | 0.50 | 0.557 | 0.815 | 0.662 | 0.512-0.772 | 44/35/10 | 32 | 32 | 0 | 0 | 2418 | - | 30.8 | 24.5 | 34.3 |
| gliner-bi | knowledgator/gliner-bi-small-v2.0@3d74c1b | open | 0.50 | 0.622 | 0.519 | 0.566 | 0.429-0.707 | 28/17/26 | 32 | 32 | 0 | 0 | 1167 | - | 20.2 | 16.1 | 21.1 |
| gliner-bi-cached | knowledgator/gliner-bi-small-v2.0@3d74c1b | open | 0.50 | 0.622 | 0.519 | 0.566 | 0.429-0.707 | 28/17/26 | 32 | 32 | 0 | 0 | 1285 | 121 | 15.2 | 11.8 | 19.7 |
| gliner-multi | urchade/gliner_multi-v2.1@443d26d | open | 0.50 | 0.564 | 0.815 | 0.667 | 0.545-0.768 | 44/34/10 | 32 | 32 | 0 | 0 | 6592 | - | 43.5 | 39.3 | 49.8 |
| gliner2 | fastino/gliner2.5-base-v1@54785d5 | open | 0.50 | 0.484 | 0.815 | 0.607 | 0.463-0.731 | 44/47/10 | 32 | 32 | 0 | 0 | 3001 | - | 41.5 | 34.8 | 44.3 |

| Per-label F1 | person | organization | location |
| --- | --- | --- | --- |
| rules | 0.000 | 0.000 | 0.000 |
| encoder | 0.800 | 0.679 | 0.605 |
| gliner | 0.533 | 0.607 | 0.622 |
| gliner25 | 0.593 | 0.645 | 0.727 |
| gliner-bi | 0.696 | 0.400 | 0.667 |
| gliner-bi-cached | 0.696 | 0.400 | 0.667 |
| gliner-multi | 0.516 | 0.689 | 0.750 |
| gliner2 | 0.621 | 0.588 | 0.625 |

Scored = completed with only valid spans. Failed and missing documents stay in the denominator. F1 95% CI = document bootstrap, 1,000 resamples, the same resamples for every model. First call = warmup on the first document.

The 95% intervals span about 0.25-0.3 F1, which is why these 32 sentences are an
integration check and not a ranking.

### Compared with the article's historical numbers

The article (updated 2026-10-02) and the previous README reported, for the same
32 sentences at threshold 0.5:

| Run | Historical TP/FP/FN | Now TP/FP/FN | Why |
| --- | --- | --- | --- |
| `gliner` | 39/38/15 | 39/38/15 | Reproduced with `gliner` 0.2.29 |
| `gliner25` | 44/35/10 | 44/35/10 | Reproduced with `gliner` 0.2.29 |
| `encoder` | 38/42/16 | 42/28/12 | Aggregation changed from `simple` to `first` |
| `rules` | 0/0/54 | 0/0/54 | Reproduced |

The bi-encoder, multilingual, and GLiNER2.5 runs are new in 0.3.0.

## ONNX FP32 and INT8 (`gliner` baseline only)

`onnx_check` on the 32 test sentences, threshold 0.5, one warmup, three repeats:

| Path | Package | p50 latency per repeat | TP/FP/FN | Micro P | Micro R |
| --- | --- | --- | --- | --- | --- |
| PyTorch | - | 37.2, 37.0, 36.6 ms | 39/38/15 | 0.506 | 0.722 |
| ONNX FP32 | 790,362,476 B | 23.4, 21.8, 21.7 ms | 39/38/15 | 0.506 | 0.722 |
| ONNX INT8 | 353,299,069 B | 16.8, 15.9, 15.7 ms | 39/41/15 | 0.487 | 0.722 |

FP32 returned exactly the PyTorch spans. INT8 changed the spans in 4 of 32
sentences and added 3 false positives, within the 0.02 recall budget. Package
sizes and span differences match the article's run. Its latencies (PyTorch 64-66
ms, FP32 16-19 ms, INT8 13-15 ms) were measured on an Apple M5 Max and are not
comparable with this machine's.

## Paid LLM baseline (`gpt-6-luna`)

Run on 2026-10-22 against `api.openai.com` with the README command, on the same
32 test sentences as the small configuration (the CLI caps the paid baseline at
32 documents). Every request finished normally and the API reported the model as
`gpt-6-luna`.

Data: DFKI-SLT/few-nerd@205f3e9 split=test; 32 documents; labels: person, organization, location
Run: Apple M4, 10 logical CPUs, torch threads -, device cpu, batch 1, 0 warmup, 1 repeats; git 5b3c7da
Thresholds: fixed 0.5

| Model | Checkpoint | Labels | Threshold | P | R | F1 | F1 95% CI | TP/FP/FN | Attempted | Scored | Failed | Missing | Load ms | Label cache ms | First call ms | p50 ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llm | gpt-6-luna | open | - | 0.057 | 0.074 | 0.065 | 0.013-0.141 | 4/66/50 | 32 | 8 | 0 | 0 | 281 | - | - | 1841.5 | 3687.9 |

| Per-label F1 | person | organization | location |
| --- | --- | --- | --- |
| llm | 0.080 | 0.034 | 0.100 |

Scored = completed with only valid spans. Failed and missing documents stay in the denominator. F1 95% CI = document bootstrap, 1,000 resamples, the same resamples for every model. First call = warmup on the first document.

The model recognized many of the right entities but miscounted their positions.
It returned 70 spans; 5 had offsets that matched their text and 65 were rejected
(63 because `text[start:end]` did not equal the returned text, 2 because the
offsets were out of range). The text of all 65 rejected spans occurs in the
sentence, and 40 of them name a gold entity with the gold label. The returned
start was off by a median of 2 characters (range -27 to +52). The scorer does not
search for the text: each rejected span is a false positive and marks its
sentence as not scored, which is why only 8 of 32 sentences were scored. This
F1 measures exact spans with model-written offsets, not entity recognition
alone.

Cost: 5,528 input and 1,681 output tokens, no reasoning tokens, no cache writes;
about USD 0.0014 at the recorded price basis. The single `ner_demo.teacher`
request used 157 input and 46 output tokens and had the same problem: "NVIDIA"
matched, and "Arm" came back at characters 41-44 instead of 45-48.

## English, German, and Russian functional examples

`python -m ner_demo.quickstart --model <name> --examples`: gold spans found out of
five per sentence (exact offsets and label).

| Run | English | German | Russian | Notes |
| --- | --- | --- | --- | --- |
| `gliner` | 4 | 4 | 4 | Misses the repeated short "Merkel" |
| `gliner25` | 4 | 5 | 4 | |
| `gliner-bi` | 5 | 5 | 4 | |
| `gliner-multi` | 4 | 4 | 4 | Misses the repeated short "Merkel"/"Меркель" |
| `gliner2` | 4 | 4 | 4 | In Russian, returns only the surname of "Ангела Меркель" |
| `encoder` | 5 | 3 | 3 | Labels German "Später" as a person and the Russian name as an organization |

All returned offsets were valid Python character offsets in the Cyrillic and
umlaut text. Three sentences test the adapters, not multilingual accuracy.

## Streaming PII demo

`python -m ner_demo.streaming` released
`Contact ████████ at ████████████████████.` with a release delay of 405 ms,
which covers the whole input plus the final pass. The opt-in tests also passed
whitespace-only chunks and the over-the-word-limit rejection with the real
checkpoint. PII recall was not measured.

## Not run

- GPU or MPS devices, Linux or Windows hosts, batching other than 1, and inputs
  longer than one sentence.
- Fine-tuning: the training converter was checked with GLiNER's real collator;
  no model was trained.
