"""Offline contract tests: fakes and mocks only, no weights or paid calls.

Real-model checks are opt-in and live in test_real_models.py.
"""

import contextlib
import copy
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ner_demo import evaluate as evaluation
from ner_demo.core import (
    LABELS,
    digest,
    example_documents,
    f1_interval,
    score,
    smoke_documents,
    validate_documents,
    validate_span,
)
from ner_demo.data import convert_row, select
from ner_demo.evaluate import evaluate
from ner_demo.streaming import BufferedPII
from ner_demo.teacher import annotate, validate_annotation
from ner_demo.training import convert_reviewed


def row(doc, entities=None, status="ok"):
    return dict(
        document_id=doc["document_id"],
        status=status,
        entities=doc["entities"] if entities is None else entities,
    )


class Contracts(unittest.TestCase):
    def test_occurrences_and_missing_documents(self):
        docs = smoke_documents()
        m = score(docs[:1], [row(docs[0], docs[0]["entities"][:1])], LABELS)
        self.assertEqual(m["micro"]["recall"], 0.5)
        self.assertEqual(score(docs, [], LABELS)["micro"]["fn"], 5)
        self.assertIsNone(score(docs, [], LABELS)["headline"])
        wrong = dict(docs[0]["entities"][0], start=1, end=5)
        m = score(docs[:1], [row(docs[0], [wrong])], LABELS)
        self.assertEqual((m["micro"]["fp"], m["micro"]["fn"]), (1, 2))

    def test_document_counts(self):
        docs = smoke_documents()
        m = score(docs, [row(docs[0]), row(docs[1], [], "error")], LABELS)
        self.assertEqual(
            [m[f"{k}_documents"] for k in ("planned", "attempted", "completed")],
            [3, 2, 1],
        )
        self.assertEqual((m["failed_documents"], m["missing_documents"]), (1, 1))
        self.assertEqual(m["valid_documents"], 1)
        self.assertIsNone(m["headline"])

    def test_f1_interval_brackets_the_point_estimate(self):
        docs = smoke_documents()  # includes a document with no entities
        self.assertEqual(f1_interval(docs, [row(d) for d in docs], LABELS), (1, 1))
        partial = [row(docs[0], docs[0]["entities"][:1]), row(docs[1])]
        low, high = f1_interval(docs, partial, LABELS)
        self.assertLessEqual(low, score(docs, partial, LABELS)["micro"]["f1"])
        self.assertLessEqual(score(docs, partial, LABELS)["micro"]["f1"], high)
        self.assertEqual(f1_interval(docs, partial, LABELS), (low, high))
        self.assertIsNone(f1_interval(docs[2:], [row(docs[2])], LABELS))

    def test_examples_have_exact_offsets(self):
        docs = example_documents()
        self.assertEqual(len(validate_documents(docs, LABELS)), 3)
        russian = docs[2]
        self.assertEqual(russian["text"][34:41], "Мюнхене")

    def test_duplicates_extra_ids_and_unknown_labels(self):
        doc = smoke_documents()[0]
        m = score([doc], [row(doc, doc["entities"] * 2)], LABELS)
        self.assertEqual(m["micro"]["fp"], 2)
        for records in [
            [row(doc), row(doc)],
            [dict(document_id="extra", status="ok", entities=[])],
        ]:
            with self.assertRaises(ValueError):
                score([doc], records, LABELS)
        m = score(
            [doc], [row(doc, [dict(doc["entities"][0], label="nonsense")])], LABELS
        )
        self.assertEqual(m["micro"]["fp"], 1)
        self.assertIsNone(m["headline"])

    def test_unicode_types_nested_empty(self):
        text = "🙂 Zoë, New York!"
        entities = [
            dict(start=2, end=5, text="Zoë", label="person"),
            dict(start=7, end=15, text="New York", label="location"),
            dict(start=11, end=15, text="York", label="organization"),
        ]
        doc = dict(document_id="x", text=text, entities=entities)
        self.assertEqual(score([doc], [row(doc)], LABELS)["micro"]["f1"], 1)
        doc["entities"] = [dict(start=2, end=5, text="Zoë", label="organization")]
        self.assertEqual(
            score([doc], [row(doc, entities[:1])], LABELS)["micro"]["f1"], 0
        )
        empty = smoke_documents()[2]
        self.assertEqual(score([empty], [row(empty)], LABELS)["micro"]["f1"], 0)
        with self.assertRaises(ValueError):
            validate_span(
                text, dict(start=True, end=5, text="Zoë", label="person"), LABELS
            )

    def test_all_pairs_survive_failure(self):
        def failure(text):
            raise RuntimeError("contains private source: do not log")

        result = evaluate(smoke_documents(), LABELS, lambda: failure, repeats=2)
        self.assertEqual(len(result["runs"]), 2)
        self.assertEqual(len(result["runs"][0]["records"]), 3)
        self.assertIsNone(result["runs"][0]["metrics"]["headline"])
        self.assertNotIn("private source", json.dumps(result))
        result = evaluate(smoke_documents(), LABELS, lambda: 1 / 0)
        metrics = result["runs"][0]["metrics"]
        self.assertEqual(metrics["completion"], 0)
        # A model that never loaded was never called: missing, not failed.
        self.assertEqual(
            (metrics["attempted_documents"], metrics["missing_documents"]), (0, 3)
        )
        self.assertEqual(result["initialization_error"], "ZeroDivisionError")

    def test_teacher_rejects_ambiguity_and_partial_output(self):
        text = "John met John"
        cases = [
            {"entities": [{"text": "John", "label": "person"}]},
            {"entities": [dict(start=0, end=4, text="Jane", label="person")]},
            {"entities": [dict(start=0, end=4, text="John", label="alien")]},
        ]
        for payload in cases:
            self.assertTrue(validate_annotation(text, payload, LABELS, "stop")[1])
        self.assertTrue(
            validate_annotation(text, {"entities": []}, LABELS, "length")[1]
        )

    def test_real_sdk_request_and_provenance(self):
        try:
            import httpx2
            from openai import OpenAI
        except ImportError:
            self.skipTest("install teacher extra to verify real SDK serialization")
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            return httpx2.Response(
                200,
                json={
                    "id": "test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "returned-snapshot",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": '{"entities": []}',
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            )

        with OpenAI(
            api_key="test-placeholder",
            http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
        ) as client:
            result = annotate(client, smoke_documents()[2])
        self.assertEqual(result["actual_model"], "returned-snapshot")
        self.assertFalse(result["fallback"])
        self.assertFalse(result["reviewed"])
        self.assertEqual(requests[0]["response_format"]["type"], "json_schema")
        self.assertEqual(requests[0]["reasoning_effort"], "none")
        self.assertNotIn("temperature", requests[0])

        def no_credits(request):
            error = dict(
                message="You have no credits remaining.",
                type="insufficient_quota",
                param=None,
                code="credit_balance_exhausted",
            )
            return httpx2.Response(429, json={"error": error})

        transport = httpx2.MockTransport(no_credits)
        with OpenAI(
            api_key="test-placeholder",
            max_retries=0,
            http_client=httpx2.Client(transport=transport),
        ) as client:
            refused = annotate(client, smoke_documents()[2])
        self.assertEqual(refused["status"], "error")
        self.assertEqual(
            refused["rejected"],
            [{"reason": "RateLimitError: credit_balance_exhausted"}],
        )
        failed = annotate(
            SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(create=lambda **kw: 1 / 0)
                )
            ),
            smoke_documents()[0],
        )
        self.assertIsNone(failed["actual_producer"])
        self.assertFalse(failed["fallback"])
        self.assertEqual(failed["status"], "error")

    def test_dataset_conversion(self):
        doc = convert_row(
            dict(
                tokens=["Zoë", ",", "New", "York"],
                ner_tags=[7, 0, 4, 4],
                fine_ner_tags=[55, 0, 21, 21],
            ),
            "test",
            1,
        )
        self.assertEqual(doc["text"], "Zoë , New York")
        self.assertEqual(doc["entities"][1]["text"], "New York")
        with self.assertRaises(ValueError):
            convert_row(dict(tokens=["a"], ner_tags=[], fine_ner_tags=[]), "test", 0)

    def test_dataset_selection_skips_texts_from_other_splits(self):
        rows = [
            dict(tokens=[word], ner_tags=[0], fine_ner_tags=[0])
            for word in ["shared", "a", "shared", "b", "c"]
        ]
        docs, skipped = select(rows, "test", 2, exclude={"shared"})
        self.assertEqual([d["text"] for d in docs], ["a", "b"])
        self.assertEqual(skipped, 2)
        self.assertEqual(docs[1]["document_id"], "few-nerd:supervised:test:3")

    def test_training_rejects_unreviewed_and_leaks(self):
        import re

        model = SimpleNamespace(
            config=SimpleNamespace(max_len=512, max_width=12),
            data_processor=SimpleNamespace(
                words_splitter=lambda text: [
                    (m.group(), m.start(), m.end())
                    for m in re.finditer(r"\w+|[^\w\s]", text)
                ]
            ),
        )
        doc = copy.deepcopy(smoke_documents()[0])
        with self.assertRaises(ValueError):
            convert_reviewed([doc], model)
        doc.update(
            reviewed=True,
            reviewer="fixture reviewer",
            status="ok",
            rejected=[],
            split="train",
            source_sha256=digest(doc["text"]),
        )
        converted = convert_reviewed([doc], model)
        self.assertEqual(
            converted["train"][0]["ner"], [[0, 0, "person"], [2, 2, "person"]]
        )
        with self.assertRaises(ValueError):
            convert_reviewed([doc, dict(doc, document_id="other", split="test")], model)
        doc["entities"][0] = dict(start=1, end=4, text="ohn", label="person")
        with self.assertRaises(ValueError):
            convert_reviewed([doc], model)

    def test_onnx_missing_package(self):
        try:
            import onnx  # noqa: F401
        except ImportError:
            self.skipTest("install onnx extra")
        from ner_demo.onnx_check import validate_package

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "model.onnx").touch()
            with self.assertRaisesRegex(ValueError, "missing gliner_config"):
                validate_package(tmp)


class FakeStream:
    """Mimics the session API, including its skip of whitespace-only chunks."""

    def __init__(self):
        self.sessions, self.cleared = {}, []

    def predict_entities(self, text, labels, threshold):
        if "Jane Doe" not in text:
            return []
        start = text.index("Jane Doe")
        return [dict(start=start, end=start + 8, text="Jane Doe", label="person")]

    def inference(self, chunks, labels, session_id, threshold):
        key = session_id[0]
        if not chunks[0].strip():
            return [[]]
        self.sessions[key] = self.sessions.get(key, "") + chunks[0]
        return [self.predict_entities(self.sessions[key], labels, threshold)]

    def clear_session(self, key):
        self.sessions.pop(key, None)
        self.cleared.append(key)


class Streaming(unittest.TestCase):
    def test_boundaries_isolation_flush(self):
        model = FakeStream()
        with BufferedPII(model, ["person"]) as a, BufferedPII(model, ["person"]) as b:
            self.assertEqual(a.push("🙂 Jane"), "")
            self.assertEqual(b.push("Public"), "")
            self.assertEqual(a.push(" Doe!"), "")
            self.assertEqual(a.finish(), "🙂 ████████!")
            self.assertEqual(b.finish(), "Public")
            self.assertGreaterEqual(a.release_delay_ms, 0)
        self.assertFalse(model.sessions)
        self.assertEqual(len(model.cleared), 2)

    def test_whitespace_chunk_keeps_offsets_aligned(self):
        # Sent alone, " " would be skipped and later offsets would shift by one.
        model = FakeStream()
        with BufferedPII(model, ["person"]) as session:
            for chunk in ["Hi", " ", "Jane Doe", "\n"]:
                self.assertEqual(session.push(chunk), "")
            self.assertEqual(session.finish(), "Hi ████████\n")
        self.assertFalse(model.sessions)

    def test_word_limit_fails_closed(self):
        # Few tokens but many words: the final GLiNER pass would truncate.
        model = FakeStream()
        model.config = SimpleNamespace(max_len=5)
        model.data_processor = SimpleNamespace(
            words_splitter=lambda text: re.finditer(r"\w+|\S", text),
            transformer_tokenizer=SimpleNamespace(encode=lambda text: [0]),
        )
        session = BufferedPII(model, ["person"])
        with self.assertRaisesRegex(ValueError, "word limit"):
            session.push("a.a.a. Jane Doe")
        self.assertTrue(session.closed)
        self.assertEqual((session.text, session.pending), ("", ""))

    def test_error_cancel_and_overflow_cleanup(self):
        for error in (RuntimeError, KeyboardInterrupt):
            model = FakeStream()
            with self.assertRaises(error):
                with BufferedPII(model, ["person"]) as session:
                    session.push("Jane")
                    raise error()
            self.assertFalse(model.sessions)
            self.assertEqual(session.text, "")
        model = FakeStream()
        with BufferedPII(model, ["person"], max_chars=2) as session:
            with self.assertRaises(ValueError):
                session.push("Jane")
            self.assertTrue(session.closed)
        self.assertEqual(len(model.cleared), 1)


def fake_runner(name, labels, download, device, threshold):
    """Predicts one extra span below 0.5 and drops one span above 0.6."""
    if name == "broken":
        raise OSError("checkpoint not in the local cache")
    entities = {
        "John met John.": [
            dict(start=0, end=4, text="John", label="person", score=0.9),
            dict(start=9, end=13, text="John", label="person", score=0.55),
            dict(start=5, end=8, text="met", label="person", score=0.4),
        ]
    }
    return lambda text: [e for e in entities.get(text, []) if e["score"] >= threshold]


class EvaluationProtocol(unittest.TestCase):
    """CLI rules that keep threshold choice off the test documents."""

    def run_cli(self, argv):
        gliner = evaluation.MODELS["gliner"]
        with (
            mock.patch.object(sys, "argv", ["evaluate", *argv.split()]),
            mock.patch.object(evaluation, "make_runner", fake_runner),
            mock.patch.dict(evaluation.MODELS, fake=gliner, broken=gliner),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            evaluation.main()

    def bundle(self, folder, name, documents, split):
        path = Path(folder, name)
        manifest = dict(split=split, labels=LABELS, documents_sha256=digest(documents))
        path.write_text(json.dumps(dict(manifest=manifest, documents=documents)))
        return path

    def test_sweep_selects_on_development_data_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            dev = self.bundle(tmp, "dev.json", smoke_documents()[:1], "validation")
            test = self.bundle(tmp, "test.json", smoke_documents()[1:], "test")
            sweep, out = Path(tmp, "sweep.json"), Path(tmp, "out.json")
            self.run_cli(
                f"--dataset {dev} --models fake rules broken "
                f"--sweep 0.3 0.5 0.7 --output {sweep}"
            )
            selected = json.loads(sweep.read_text())
            # "broken" never loaded, so it exports no threshold.
            self.assertEqual(selected["selected_thresholds"], {"fake": 0.5})
            self.assertEqual(len(selected["results"]["fake"]["sweep"]), 3)
            self.run_cli(
                f"--dataset {test} --models fake --thresholds-from {sweep} "
                f"--output {out}"
            )
            models = json.loads(out.read_text())["manifest"]["models"]
            self.assertEqual(models["fake"]["threshold"], 0.5)
            refusals = [
                # Thresholds may not be chosen on the test split ...
                f"--dataset {test} --sweep 0.5",
                # ... nor reused on the documents they were chosen on ...
                f"--dataset {dev} --thresholds-from {sweep}",
                # ... and every thresholded model needs a development threshold.
                f"--dataset {test} --models broken --thresholds-from {sweep}",
            ]
            for argv in refusals:
                with self.assertRaises(SystemExit):
                    self.run_cli(f"{argv} --output {out}")

    def test_provenance_without_git(self):
        with mock.patch.object(evaluation.subprocess, "run", side_effect=OSError):
            state = evaluation.git_state()
        self.assertEqual(state, dict(git_commit=None, git_dirty=None))


class Adapters(unittest.TestCase):
    def test_gliner2_keeps_repeats_and_native_scores(self):
        from ner_demo.models import gliner2_predict

        calls = []

        def extract_entities(text, labels, **options):
            calls.append(options)
            found = [
                dict(text="John", start=0, end=4, confidence=0.9),
                dict(text="John", start=9, end=13, confidence=0.8),
            ]
            return {"entities": {"person": found, "location": []}}

        model = SimpleNamespace(
            config=SimpleNamespace(max_len=8),
            processor=SimpleNamespace(
                word_splitter=lambda text: re.finditer(r"\w+|\S", text)
            ),
            extract_entities=extract_entities,
        )
        spans = gliner2_predict(model, "John met John.", ["person", "location"], 0.4)
        self.assertEqual(
            [(e["start"], e["score"]) for e in spans], [(0, 0.9), (9, 0.8)]
        )
        self.assertTrue(calls[0]["include_spans"] and calls[0]["include_confidence"])
        self.assertEqual(calls[0]["threshold"], 0.4)
        with self.assertRaisesRegex(ValueError, "word limit"):
            gliner2_predict(
                model, "one two three four five six seven eight nine", ["person"]
            )

    def test_registry_table_is_in_readme(self):
        from ner_demo.registry import MODELS, table

        for name, entry in MODELS.items():
            revision = entry["revision"]
            self.assertTrue(
                revision is None or re.fullmatch(r"[0-9a-f]{40}", revision), name
            )
            self.assertIn(
                entry["status"], {"validated", "contract", "implemented", "blocked"}
            )
        readme = (Path(__file__).parents[1] / "README.md").read_text()
        self.assertTrue(
            table() in readme, "stale README table: python -m ner_demo.registry"
        )


if __name__ == "__main__":
    unittest.main()
