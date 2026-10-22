# Results recorded in this repository

Every number on this page was produced on **2026-10-22** by the commands in the
README, on one machine: Apple M4 (10 logical CPUs; torch used 4 threads), macOS
27.2, CPU only, batch size 1, Python 3.13.3, `gliner` 0.2.29, `gliner2` 2.1.0,
transformers 5.1.0, torch 2.11.0, ONNX Runtime 1.24.4. Models loaded offline
(`HF_HUB_OFFLINE=1`) from the pinned revisions. The runs were made from the
working tree that became release 0.3.0, so the outputs record `git_dirty: true`
against the previous commit.

The task is Few-NERD's three coarse types (person, organization, location) on
English Wikipedia sentences. These results check the adapters under one set of
rules; they are not a general ranking of the model families. No model-card
scores are copied here.

## Larger configuration

### Thresholds chosen on 500 validation sentences

Data: DFKI-SLT/few-nerd@205f3e9 split=validation; 500 documents; labels: person, organization, location
Run: Apple M4, 10 logical CPUs, torch threads 4, device cpu, batch 1, 0 warmup, 1 repeats; git 18c9b92 (dirty)
Thresholds: development sweep: 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97

| Model | Checkpoint | Labels | Threshold | P | R | F1 | F1 95% CI | TP/FP/FN | Attempted | Scored | Failed | Missing | Load ms | Label cache ms | First call ms | p50 ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gliner | urchade/gliner_medium-v2.1@40ec419 | open | 0.70 | 0.638 | 0.691 | 0.663 | - | 660/375/295 | 500 | 500 | 0 | 0 | 3261 | - | - | 37.1 | 49.7 |
| gliner25 | gliner-community/gliner_small-v2.5@f227d3c | open | 0.80 | 0.690 | 0.762 | 0.724 | - | 728/327/227 | 500 | 500 | 0 | 0 | 2360 | - | - | 24.8 | 34.6 |
| gliner-bi | knowledgator/gliner-bi-small-v2.0@3d74c1b | open | 0.50 | 0.746 | 0.725 | 0.735 | - | 692/235/263 | 500 | 500 | 0 | 0 | 1120 | - | - | 16.8 | 23.9 |
| gliner-bi-cached | knowledgator/gliner-bi-small-v2.0@3d74c1b | open | 0.50 | 0.746 | 0.725 | 0.735 | - | 692/235/263 | 500 | 500 | 0 | 0 | 1245 | 120 | - | 12.1 | 17.9 |
| gliner-multi | urchade/gliner_multi-v2.1@443d26d | open | 0.80 | 0.668 | 0.677 | 0.673 | - | 647/321/308 | 500 | 500 | 0 | 0 | 5641 | - | - | 39.5 | 52.3 |
| gliner2 | fastino/gliner2.5-base-v1@54785d5 | open | 0.95 | 0.699 | 0.685 | 0.692 | - | 654/282/301 | 500 | 500 | 0 | 0 | 2429 | - | - | 35.2 | 45.5 |

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
Run: Apple M4, 10 logical CPUs, torch threads 4, device cpu, batch 1, 1 warmup, 1 repeats; git 18c9b92 (dirty)
Thresholds: from artifacts/thresholds-validation-500.json, chosen on DFKI-SLT/few-nerd split=validation

| Model | Checkpoint | Labels | Threshold | P | R | F1 | F1 95% CI | TP/FP/FN | Attempted | Scored | Failed | Missing | Load ms | Label cache ms | First call ms | p50 ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rules | - | fixed | - | 0.167 | 0.001 | 0.003 | 0.001-0.005 | 5/25/3774 | 2000 | 2000 | 0 | 0 | 0 | - | 0.2 | 0.0 | 0.0 |
| encoder | dslim/bert-base-NER@d1a3e8f | fixed | - | 0.635 | 0.747 | 0.687 | 0.666-0.705 | 2823/1621/956 | 2000 | 2000 | 0 | 0 | 1740 | - | 470.6 | 16.9 | 23.7 |
| gliner | urchade/gliner_medium-v2.1@40ec419 | open | 0.70 | 0.613 | 0.665 | 0.638 | 0.618-0.659 | 2513/1584/1266 | 2000 | 2000 | 0 | 0 | 3514 | - | 56.2 | 37.0 | 48.0 |
| gliner25 | gliner-community/gliner_small-v2.5@f227d3c | open | 0.80 | 0.656 | 0.740 | 0.695 | 0.677-0.713 | 2795/1464/984 | 2000 | 2000 | 0 | 0 | 2429 | - | 30.9 | 24.5 | 32.8 |
| gliner-bi | knowledgator/gliner-bi-small-v2.0@3d74c1b | open | 0.50 | 0.729 | 0.675 | 0.701 | 0.681-0.719 | 2551/949/1228 | 2000 | 2000 | 0 | 0 | 1158 | - | 20.3 | 16.3 | 22.9 |
| gliner-bi-cached | knowledgator/gliner-bi-small-v2.0@3d74c1b | open | 0.50 | 0.729 | 0.675 | 0.701 | 0.681-0.719 | 2551/949/1228 | 2000 | 2000 | 0 | 0 | 1255 | 120 | 15.8 | 11.9 | 16.8 |
| gliner-multi | urchade/gliner_multi-v2.1@443d26d | open | 0.80 | 0.641 | 0.636 | 0.639 | 0.618-0.660 | 2403/1344/1376 | 2000 | 2000 | 0 | 0 | 6718 | - | 43.0 | 39.7 | 50.7 |
| gliner2 | fastino/gliner2.5-base-v1@54785d5 | open | 0.95 | 0.687 | 0.649 | 0.668 | 0.651-0.686 | 2454/1117/1325 | 2000 | 2000 | 0 | 0 | 3011 | - | 42.3 | 35.0 | 44.3 |

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
| 2,000-sentence run, p50 / p95 | 16.3 / 22.9 ms | 11.9 / 16.8 ms |
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
Run: Apple M4, 10 logical CPUs, torch threads 4, device cpu, batch 1, 1 warmup, 3 repeats; git 18c9b92 (dirty)
Thresholds: fixed 0.5

| Model | Checkpoint | Labels | Threshold | P | R | F1 | F1 95% CI | TP/FP/FN | Attempted | Scored | Failed | Missing | Load ms | Label cache ms | First call ms | p50 ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rules | - | fixed | - | 0.000 | 0.000 | 0.000 | 0.000-0.000 | 0/0/54 | 32 | 32 | 0 | 0 | 0 | - | 0.1 | 0.0 | 0.0 |
| encoder | dslim/bert-base-NER@d1a3e8f | fixed | - | 0.600 | 0.778 | 0.677 | 0.551-0.780 | 42/28/12 | 32 | 32 | 0 | 0 | 1763 | - | 282.8 | 17.1 | 22.2 |
| gliner | urchade/gliner_medium-v2.1@40ec419 | open | 0.50 | 0.506 | 0.722 | 0.595 | 0.439-0.722 | 39/38/15 | 32 | 32 | 0 | 0 | 3455 | - | 57.3 | 36.8 | 48.2 |
| gliner25 | gliner-community/gliner_small-v2.5@f227d3c | open | 0.50 | 0.557 | 0.815 | 0.662 | 0.512-0.772 | 44/35/10 | 32 | 32 | 0 | 0 | 2427 | - | 29.6 | 24.6 | 33.8 |
| gliner-bi | knowledgator/gliner-bi-small-v2.0@3d74c1b | open | 0.50 | 0.622 | 0.519 | 0.566 | 0.429-0.707 | 28/17/26 | 32 | 32 | 0 | 0 | 1172 | - | 22.2 | 16.1 | 21.2 |
| gliner-bi-cached | knowledgator/gliner-bi-small-v2.0@3d74c1b | open | 0.50 | 0.622 | 0.519 | 0.566 | 0.429-0.707 | 28/17/26 | 32 | 32 | 0 | 0 | 1251 | 121 | 15.5 | 12.1 | 20.0 |
| gliner-multi | urchade/gliner_multi-v2.1@443d26d | open | 0.50 | 0.564 | 0.815 | 0.667 | 0.545-0.768 | 44/34/10 | 32 | 32 | 0 | 0 | 6595 | - | 43.2 | 38.2 | 50.9 |
| gliner2 | fastino/gliner2.5-base-v1@54785d5 | open | 0.50 | 0.484 | 0.815 | 0.607 | 0.463-0.731 | 44/47/10 | 32 | 32 | 0 | 0 | 2987 | - | 42.9 | 34.8 | 44.2 |

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
| PyTorch | - | 37.2, 36.9, 37.4 ms | 39/38/15 | 0.506 | 0.722 |
| ONNX FP32 | 790,362,476 B | 22.8, 20.9, 19.8 ms | 39/38/15 | 0.506 | 0.722 |
| ONNX INT8 | 353,299,069 B | 16.0, 14.9, 15.9 ms | 39/41/15 | 0.487 | 0.722 |

FP32 returned exactly the PyTorch spans. INT8 changed the spans in 4 of 32
sentences and added 3 false positives, within the 0.02 recall budget. Package
sizes and span differences match the article's run. Its latencies (PyTorch 64-66
ms, FP32 16-19 ms, INT8 13-15 ms) were measured on an Apple M5 Max and are not
comparable with this machine's.

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
`Contact ████████ at ████████████████████.` with a release delay of 403 ms,
which covers the whole input plus the final pass. The opt-in tests also passed
whitespace-only chunks and the over-the-word-limit rejection with the real
checkpoint. PII recall was not measured.

## Not run

- The paid OpenAI baseline (`llm`) and the teacher command: no paid requests were
  made. Offline tests cover the request and the response handling through the
  real SDK with a mock transport.
- GPU or MPS devices, Linux or Windows hosts, batching other than 1, and inputs
  longer than one sentence.
- Fine-tuning: the training converter was checked with GLiNER's real collator;
  no model was trained.
