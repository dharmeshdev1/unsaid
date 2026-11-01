#!/usr/bin/env python3
"""Turn a Wikipedia dump into the JSONL file that `unsaid build` reads.

Download the dump from Hugging Face, then:

    uv run --with pyarrow scripts/prepare_wikipedia.py simple.parquet -o wiki.jsonl
    uv run --with pyarrow scripts/prepare_wikipedia.py simple.parquet -o small.jsonl \
        --top 20000 --max-passages 1

Dump: https://huggingface.co/datasets/wikimedia/wikipedia (the 20231101.simple configuration).
Wikipedia text is licensed CC BY-SA 4.0, and every passage keeps the URL of its article.
"""
import argparse
import json
import re

import pyarrow.parquet as pq

WORDS = 120
OVERLAP = 20


def passages(text, limit):
    """Split like `unsaid build` does, so a second pass there leaves the passages unchanged."""
    words = text.split()
    out, start = [], 0
    while words and len(out) < limit:
        end = min(start + WORDS, len(words))
        out.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += WORDS - OVERLAP
    return out


def worth_keeping(title, text):
    """Leave out pages that are lists of links rather than prose about something."""
    if len(text) < 80 or title.isdigit() or title.lower().startswith(("list of", "lists of")):
        return False
    if re.search(r"may (also )?refer to|can (also )?(mean|refer to)", text[:200]):
        return False
    return not re.fullmatch(r"(January|February|March|April|May|June|July|August|September"
                            r"|October|November|December) \d{1,2}", title)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dump", help="parquet file with title, text and url columns")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("--top", type=int, help="keep only the N longest articles")
    parser.add_argument("--max-passages", type=int, default=3, help="most passages per article")
    args = parser.parse_args()

    table = pq.read_table(args.dump, columns=["title", "text", "url"])
    rows = [
        (title, text, url)
        for title, text, url in zip(*(table.column(c).to_pylist() for c in table.column_names))
        if worth_keeping(title, text)
    ]
    if args.top:
        # Length is a rough stand-in for how notable a subject is. Ties keep dump order.
        rows.sort(key=lambda row: -len(row[1]))
        rows = sorted(rows[: args.top], key=lambda row: row[0])

    count = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for title, text, url in rows:
            for passage in passages(text, args.max_passages):
                out.write(json.dumps({"title": title, "text": passage, "url": url},
                                     ensure_ascii=False) + "\n")
                count += 1
    print(f"{len(rows)} articles, {count} passages")


if __name__ == "__main__":
    main()
