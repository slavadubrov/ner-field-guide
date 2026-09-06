"""Session-isolated PII buffering. No text is released before final detection."""

import argparse
import time
import uuid

from ner_demo.core import validate_span
from ner_demo.models import load_gliner


class BufferedPII:
    """ponytail: whole-session buffer; bounded memory, full-session release delay.

    A finite chunk lookbehind cannot guarantee detection of arbitrarily long PII.
    Use a separately validated incremental release policy if latency demands it.
    This prevents premature release, not detector false negatives.
    """

    def __init__(self, model, labels, max_chars=16000):
        self.model, self.labels = model, list(labels)
        self.session_id = str(uuid.uuid4())
        self.text, self.closed = "", False
        self.max_chars = max_chars
        self.started = time.perf_counter()
        self.release_delay_ms = None

    def push(self, chunk):
        if self.closed:
            raise ValueError("session is closed")
        try:
            if (
                not isinstance(chunk, str)
                or not chunk
                or len(self.text) + len(chunk) > self.max_chars
            ):
                raise ValueError("chunk is empty, invalid, or exceeds session bound")
            self.text += chunk
            if hasattr(self.model, "data_processor"):
                processor = self.model.data_processor
                token_count = len(
                    processor.transformer_tokenizer.encode(
                        self.text + " " + " ".join(self.labels)
                    )
                )
                if token_count + 64 > self.model.config.max_len:
                    raise ValueError("session exceeds conservative model token budget")
            snapshot = self.model.inference(
                [chunk], self.labels, session_id=[self.session_id], threshold=0.5
            )[0]
            for entity in snapshot:
                validate_span(self.text, entity, self.labels)
            return ""  # Do not expose intermediate spans or source text to downstream output.
        except BaseException:
            self.close()
            raise

    def finish(self):
        if self.closed:
            raise ValueError("session is closed")
        try:
            # The API requires a nonempty new chunk for recompute. A stateless
            # complete-text pass flushes pending entities without injecting text.
            entities = (
                self.model.predict_entities(self.text, self.labels, threshold=0.5)
                if self.text
                else []
            )
            masked = list(self.text)
            for entity in entities:
                start, end, _ = validate_span(self.text, entity, self.labels)
                masked[start:end] = "█" * (end - start)
            result = "".join(masked)
            self.release_delay_ms = 1000 * (time.perf_counter() - self.started)
            return result
        finally:
            self.close()

    def close(self):
        if not self.closed:
            self.closed = True
            self.text = ""
            self.model.clear_session(self.session_id)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    model = load_gliner("streaming", args.download)
    if not callable(getattr(model, "clear_session", None)):
        raise RuntimeError("loaded checkpoint does not support streaming sessions")
    with BufferedPII(model, ["person", "email address"]) as session:
        for chunk in ["Contact Jane", " Doe at jane", ".doe@example.com."]:
            assert session.push(chunk) == ""
        print(session.finish())
        print(
            f"Release delay: {session.release_delay_ms:.1f} ms; includes entire input duration"
        )


if __name__ == "__main__":
    main()
