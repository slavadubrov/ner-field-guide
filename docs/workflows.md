# Workflows beyond the model comparison

These match sections of the
[article](https://slavadubrov.com/blog/2026/04/02/ner-guide/). Commands assume
`uv sync --locked --all-extras` has run; model commands need `--download` once.

## LLM annotations and training input (paid, opt-in)

`ner_demo.teacher` sends one Chat Completions request with a strict JSON Schema.
The model quotes each entity exactly and says which appearance of the quote it
is; the code computes the offsets from that, because models miscount characters.
A quote that does not appear as whole words, an appearance number that does not
exist, an unknown label, or a duplicate is rejected with a reason, and nothing is
searched for beyond the stated appearance. It records the requested and returned model,
prompt, settings, usage, and price basis. Records start unreviewed.

```sh
cp .env.example .env   # set OPENAI_API_KEY; optional OPENAI_BASE_URL
uv run --extra teacher python -m ner_demo.teacher --allow-paid
```

- The default model is `gpt-6-luna`, chosen for cost, not for NER accuracy.
  OpenAI publishes no dated snapshot for it, so each record saves the model ID the
  API returns. The command makes at most one request, with no retries and at most
  2,048 output tokens. It writes `artifacts/teacher_annotations.jsonl`.
- `.env` is read only by commands that pass `--allow-paid`. No paid call runs in
  the tests or in CI.
- Paid evaluation is capped at 32 documents of 4,000 characters, one repeat, and
  no warmup:

  ```sh
  uv run --extra teacher python -m ner_demo.evaluate --models llm \
    --allow-paid --repeats 1 --warmups 0
  ```

**Human review is required before training.** Review the source and every
annotation, resolve the rejected entries, and set `reviewed: true`, a nonempty
`reviewer`, `status: "ok"`, and a `split` (`train`, `development`, or `test`) in
each JSONL record. Keep `source_sha256` intact. Then convert the reviewed file:

```sh
uv run --extra models python -m ner_demo.training artifacts/reviewed.jsonl
```

This writes `artifacts/training/{train,development,test}.json` in GLiNER's
training format (`tokenized_text` plus `ner` triples with inclusive token ends).
Each row passes through GLiNER's real `SpanDataCollator`. The converter rejects
unreviewed records, unresolved rejections, edited sources, missing splits,
duplicate IDs, identical texts in different splits, boundaries that do not align
with GLiNER's word splitter, spans wider than the model allows, and anything that
would be truncated. It does not train a model.

## ONNX FP32 and INT8 (validated for `gliner` only)

```sh
uv run --extra models --extra onnx python -m ner_demo.onnx_check \
  --dataset artifacts/few-nerd-test.json --output artifacts/onnx-comparison
```

The command exports an FP32 package from `urchade/gliner_medium-v2.1`, quantizes a
copy to INT8, validates each graph, config, tokenizer, and external weight file,
and loads both through GLiNER's ONNX Runtime path. PyTorch, FP32, and INT8 run on
the same inputs, labels, threshold (0.5), and batch size (1). `comparison.json`
holds the differing documents, per-label metrics, latencies, package sizes, and
environment. The command fails when FP32 spans differ from PyTorch, or when INT8
micro recall is more than 0.02 below PyTorch. The output directory must be new;
`--reuse` re-checks existing packages.

INT8 uses dynamic quantization of `MatMul`, `Gemm`, and the embedding `Gather`,
and keeps the twelve DeBERTa FFN down-projections (`layer.N/output/dense`) in
FP32: quantizing only those twelve layers already makes this model return no
entities. This export and quantization recipe was validated only for this
checkpoint. Other registry models need their own parity and recall check before
anyone should rely on an ONNX package of them.

Results are in the README. See also
[ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

## Streaming PII buffering

```sh
uv run --extra models python -m ner_demo.streaming --download
```

[GLiNER Streaming PII](https://huggingface.co/knowledgator/gliner-stream-pii-v1.0)
has a session API that returns offsets into the accumulated text. `BufferedPII`
gives each session its own ID, keeps labels fixed, and releases nothing until a
final full-text detection pass; it then returns the text with detected
characters masked and clears the session.

Buffering contracts, each covered by offline tests with a fake session model and
by an opt-in test with the real checkpoint:

- Completion, exceptions, cancellation (`KeyboardInterrupt`), and context-manager
  exit all clear the buffered text and the model's session state.
- A chunk that would exceed the 16,000-character cap closes the session.
- A session that exceeds the model's word limit (8,192 words) is rejected before
  release. Without this check the final pass would keep only the first 8,192
  words and release the rest unscanned.
- Whitespace-only chunks are held and sent with the next chunk that has visible
  text. The session API skips whitespace-only input, which would otherwise shift
  every later offset.

These are buffering guarantees, not detection guarantees. A span the detector
misses is released unmasked; the tests check buffering behavior, not PII recall.
Release delay covers the whole input plus the final inference; a shorter delay
needs its own tested policy, because a fixed lookbehind cannot protect
arbitrarily long entities. Measure leaked characters separately from document
NER F1.
