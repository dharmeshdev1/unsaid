#!/usr/bin/env python3
"""Answer a question with a local model, retrieving its context privately.

    unsaid proxy --server URL
    python3 examples/private_rag.py "what causes tides"                     # prints the prompt
    python3 examples/private_rag.py "what causes tides" --ollama llama3.2   # answers with Ollama

A normal RAG pipeline sends your question to whoever hosts the documents. Here the document server
only ever sees ciphertext, and the model runs on your machine, so nothing reads the question except
your own computer.
"""
import argparse
import json
import urllib.request

from search import search

PROMPT = """Answer the question using only the passages below. Say so if they do not contain the answer.

{passages}

Question: {question}
Answer:"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("question")
    parser.add_argument("--ollama", metavar="MODEL", help="answer with this local Ollama model")
    parser.add_argument("-k", type=int, default=4, help="passages to retrieve")
    args = parser.parse_args()

    hits = search(args.question, args.k)["hits"]
    passages = "\n\n".join(f"[{hit['title']}] {hit['text']}" for hit in hits)
    prompt = PROMPT.format(passages=passages, question=args.question)
    if not args.ollama:
        print(prompt)
        return
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps({"model": args.ollama, "prompt": prompt, "stream": False}).encode(),
    )
    with urllib.request.urlopen(request) as response:
        print(json.load(response)["response"])


if __name__ == "__main__":
    main()
