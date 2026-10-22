"""Opt-in checks against the real pinned weights: no downloads, no paid calls.

Download the checkpoints once (see README), then run:
    NER_LOCAL_CHECKS=1 HF_HUB_OFFLINE=1 uv run --all-extras \
        python -m unittest tests.test_real_models -v
A checkpoint missing from the local cache fails the test; it is not skipped.
"""

import os
import unittest

from ner_demo.core import (
    LABELS,
    digest,
    example_documents,
    smoke_documents,
    validate_span,
)

GLINER_FAMILY = ["gliner", "gliner25", "gliner-bi", "gliner-multi", "gliner2"]


def spans(entities):
    return sorted((e["start"], e["end"], e["label"]) for e in entities)


@unittest.skipUnless(os.getenv("NER_LOCAL_CHECKS") == "1", "opt-in real models")
class RealModels(unittest.TestCase):
    def test_offsets_scores_and_repeated_mentions(self):
        from ner_demo.models import make_runner

        documents = smoke_documents() + example_documents()
        for name in [*GLINER_FAMILY, "encoder"]:
            with self.subTest(name):
                predict = make_runner(name, LABELS)
                for doc in documents:
                    for entity in predict(doc["text"]):
                        validate_span(doc["text"], entity, LABELS)
                        self.assertTrue(0 <= entity["score"] <= 1)
                if name in GLINER_FAMILY:
                    self.assertEqual(
                        spans(predict("John met John.")),
                        [(0, 4, "person"), (9, 13, "person")],
                    )

    def test_label_cache_matches_uncached_predictions(self):
        from ner_demo.models import make_runner

        uncached = make_runner("gliner-bi", LABELS)
        cached = make_runner("gliner-bi-cached", LABELS)
        self.assertIsNone(uncached.label_cache_ms)
        self.assertGreater(cached.label_cache_ms, 0)
        for doc in smoke_documents() + example_documents():
            self.assertEqual(
                spans(uncached(doc["text"])), spans(cached(doc["text"])), doc
            )

    def test_cyrillic_offsets(self):
        from ner_demo.models import make_runner

        russian = example_documents()[2]
        found = spans(make_runner("gliner-multi", LABELS)(russian["text"]))
        self.assertIn((0, 14, "person"), found)  # "Ангела Меркель"
        self.assertIn((34, 41, "location"), found)  # "Мюнхене"

    def test_long_inputs_fail_instead_of_truncating(self):
        from ner_demo.models import make_runner

        for name in GLINER_FAMILY:
            with self.subTest(name):
                predict = make_runner(name, LABELS)
                with self.assertRaises(ValueError):
                    predict("word " * 5000)
        for name in ["gliner", "gliner-bi"]:
            with self.subTest(name):
                predict = make_runner(name, ["unusually long label " * 1000])
                with self.assertRaises(ValueError):
                    predict("John")

    def test_training_rows_pass_the_real_collator(self):
        from ner_demo.models import load_gliner
        from ner_demo.training import check_collator, convert_reviewed

        model = load_gliner()
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
        self.assertEqual(check_collator(convert_reviewed(docs, model), model), 3)

    def test_streaming_session(self):
        from ner_demo.models import load_gliner
        from ner_demo.streaming import BufferedPII

        model = load_gliner("streaming")
        labels = ["person", "email address"]
        with BufferedPII(model, labels) as session:
            for chunk in ["Contact Jane", " ", "Doe at jane.doe@example.com", "\n"]:
                self.assertEqual(session.push(chunk), "")
            released = session.finish()
        self.assertNotIn("Jane Doe", released)
        self.assertNotIn("jane.doe@example.com", released)
        self.assertEqual(model.session_count, 0)
        # Under the character cap but over the word limit: rejected before
        # any inference, so the final pass cannot truncate the tail unscanned.
        session = BufferedPII(model, labels)
        with self.assertRaisesRegex(ValueError, "word limit"):
            session.push("a." * 4300 + " jane.doe@example.com")
        self.assertEqual(model.session_count, 0)


if __name__ == "__main__":
    unittest.main()
