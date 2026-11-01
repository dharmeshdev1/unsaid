#!/usr/bin/env python3
"""Embed a large JSONL corpus on a GPU, for `unsaid build --embeddings`.

`unsaid build` embeds on the CPU with no other tools installed, which is simplest but takes hours
for a million passages. This produces the same kind of file much faster:

    uv run --with sentence-transformers scripts/embed_fast.py wiki.jsonl -o wiki.f32
    unsaid build wiki.jsonl --embeddings wiki.f32

Run `unsaid build` with the same model and token limit used here. Each line of the input must
already be a single passage, which is what scripts/prepare_wikipedia.py writes.
"""
import argparse
import json
import time
from itertools import islice

import numpy as np
from sentence_transformers import SentenceTransformer

SLICE = 50_000
MODELS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "bge": ("BAAI/bge-small-en-v1.5", ""),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("corpus")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("--model", choices=MODELS, default="minilm")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    name, prefix = MODELS[args.model]
    model = SentenceTransformer(name)
    model.max_seq_length = args.max_tokens
    done, started = 0, time.time()
    with open(args.corpus, encoding="utf-8") as lines, open(args.out, "wb") as out:
        # Work through the file in slices, so a corpus of millions does not have to fit in memory.
        while slice_ := list(islice(lines, SLICE)):
            texts = []
            for line in slice_:
                row = json.loads(line)
                title, text = row.get("title", ""), " ".join(row["text"].split())
                texts.append(prefix + (f"{title}. {text}" if title else text))
            vectors = model.encode(
                texts, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=False
            )
            np.asarray(vectors, dtype="<f4").tofile(out)
            done += len(texts)
            rate = done / (time.time() - started)
            print(f"{done} passages, {rate:.0f} per second", flush=True)
    print(f"{done} vectors of {model.get_sentence_embedding_dimension()} values in {args.out}")


if __name__ == "__main__":
    main()
