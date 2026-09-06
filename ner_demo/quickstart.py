"""Cached pinned GLiNER inference; opt in before downloading weights."""

import argparse

from ner_demo.core import LABELS
from ner_demo.models import gliner_predict, load_gliner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    model = load_gliner(download=args.download)
    print(gliner_predict(model, "Bill Gates founded Microsoft.", LABELS))


if __name__ == "__main__":
    main()
