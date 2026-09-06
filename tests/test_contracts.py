"""Offline regressions for defects that previously produced false evidence."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ner_demo.core import LABELS, digest, score, smoke_documents, validate_span
from ner_demo.data import convert_row
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
        self.assertEqual(result["runs"][0]["metrics"]["completion"], 0)

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
            import httpx
            from openai import OpenAI
        except ImportError:
            self.skipTest("install teacher extra to verify real SDK serialization")
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(
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
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        ) as client:
            result = annotate(client, smoke_documents()[2])
        self.assertEqual(result["actual_model"], "returned-snapshot")
        self.assertFalse(result["fallback"])
        self.assertFalse(result["reviewed"])
        self.assertEqual(requests[0]["response_format"]["type"], "json_schema")
        self.assertEqual(requests[0]["reasoning_effort"], "none")
        self.assertNotIn("temperature", requests[0])
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
    def __init__(self):
        self.sessions, self.cleared = {}, []

    def predict_entities(self, text, labels, threshold):
        if "Jane Doe" not in text:
            return []
        start = text.index("Jane Doe")
        return [dict(start=start, end=start + 8, text="Jane Doe", label="person")]

    def inference(self, chunks, labels, session_id, threshold):
        key = session_id[0]
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


@unittest.skipUnless(
    __import__("os").getenv("NER_LOCAL_CHECKS") == "1",
    "opt-in cached model integration",
)
class LocalIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ner_demo.models import load_gliner

        cls.model = load_gliner()

    def test_actual_reader_and_positive_positions(self):
        from ner_demo.training import check_collator

        docs = smoke_documents()
        for index, doc in enumerate(docs):
            doc.update(
                reviewed=True,
                reviewer="synthetic regression fixture",
                status="ok",
                rejected=[],
                split=["train", "development", "test"][index],
                source_sha256=digest(doc["text"]),
            )
        self.assertEqual(
            check_collator(convert_reviewed(docs, self.model), self.model), 3
        )

    def test_length_rejection(self):
        from ner_demo.models import gliner_predict

        with self.assertRaises(ValueError):
            gliner_predict(self.model, "word " * 1000, LABELS)
        with self.assertRaises(ValueError):
            gliner_predict(self.model, "John", ["unusually long label " * 1000])


if __name__ == "__main__":
    unittest.main()
