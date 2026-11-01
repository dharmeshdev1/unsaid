#!/usr/bin/env python3
"""Search privately from any program, using only the standard library.

    unsaid serve index             # on the machine that holds the index
    unsaid proxy --server URL      # on your own machine
    python3 examples/search.py "how do vaccines work"

Your program talks plain JSON to the gateway on your own machine. The gateway does the embedding and
the encryption, so only ciphertext goes out.
"""
import json
import sys
import urllib.request

GATEWAY = "http://127.0.0.1:8378/search"


def search(query, limit=5):
    request = urllib.request.Request(
        GATEWAY,
        data=json.dumps({"query": query, "limit": limit}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


if __name__ == "__main__":
    result = search(" ".join(sys.argv[1:]) or "how do vaccines work")
    for hit in result["hits"]:
        print(f"{hit['score']:.2f}  {hit['title']}\n      {hit['text'][:160]}\n")
    trace = result["trace"]
    print(f"sent {trace['uploaded']} bytes, received {trace['downloaded']} bytes")
