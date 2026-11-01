#!/usr/bin/env python3
"""A complete private search client in Python, talking to a real Unsaid server.

    unsaid demo                                          # in one terminal
    python3 tutorial/3_real_client.py "why is the sky blue"

This shares no code with the Rust client, only docs/protocol.md, and that is the point: if you can
read this file, you understand everything that goes over the wire. It needs numpy and the `unsaid`
binary, which it asks for the query's embedding so that this file does not need a neural network.
It is written to be read, so unlike the Rust client it does not defend against a malicious server,
wipe its secrets, or keep its traffic to a fixed shape when things go wrong.
"""
import argparse
import hashlib
import json
import subprocess
import urllib.request
import zlib

import numpy as np

N = 1280
SIGMA = 6.4
DIGEST_BYTES = 8


def get(base, path):
    with urllib.request.urlopen(base + path) as response:
        return response.read()


def post(base, path, body):
    request = urllib.request.Request(base + path, data=body, method="POST")
    with urllib.request.urlopen(request) as response:
        return response.read()


def blob(base, manifest, name):
    """Download a public file and check it against the hash in the manifest."""
    info = manifest["blobs"][name]
    data = get(base, "/v1/blob/" + info["sha256"])
    assert hashlib.sha256(data).hexdigest() == info["sha256"], f"{name} failed its hash check"
    return data


def content_id(manifest):
    """Protocol section 2. Both public matrices are derived from this hash of the index."""
    m = manifest
    numbers = " ".join(
        f"{value:g}" if isinstance(value, float) else str(value)
        for value in [
            m["format"], m["documents"], m["model"], m["model_dim"], m["dim"], m["clusters"],
            m["prototypes"], m["doc_scale"], m["query_scale"], m["rank"]["rows"], m["rank"]["cols"],
            m["rank"]["log_p"], m["payload"]["rows"], m["payload"]["cols"], m["payload"]["log_p"],
        ]
    )
    h = hashlib.sha256(b"unsaid/v1/content\0" + numbers.encode())
    for name in ["model", "page_digests", "layout", "routing"]:
        h.update(b"\0" + name.encode() + b"=" + m["blobs"][name]["sha256"].encode())
    return h.digest()


def matrix_seed(manifest, label):
    return hashlib.sha256(
        b"unsaid/v1/matrix-a\0" + label.encode() + b"\0" + content_id(manifest)
    ).digest()


def rotate(x, bits):
    return (x << np.uint32(bits)) | (x >> np.uint32(32 - bits))


def public_matrix(seed, rows):
    """ChaCha12 from RFC 8439, vectorised. Row j is the keystream for nonce (j, j >> 32, 0)."""
    key = np.frombuffer(seed, dtype="<u4")
    blocks = N // 16
    row = np.repeat(np.arange(rows, dtype=np.uint64), blocks)
    counter = np.tile(np.arange(blocks, dtype=np.uint32), rows)
    state = np.zeros((16, rows * blocks), dtype=np.uint32)
    state[:4] = np.array([0x61707865, 0x3320646E, 0x79622D32, 0x6B206574], dtype=np.uint32)[:, None]
    state[4:12] = key[:, None]
    state[12] = counter
    state[13] = (row & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    state[14] = (row >> np.uint64(32)).astype(np.uint32)
    x = state.copy()

    def quarter(a, b, c, d):
        x[a] += x[b]; x[d] = rotate(x[d] ^ x[a], 16)
        x[c] += x[d]; x[b] = rotate(x[b] ^ x[c], 12)
        x[a] += x[b]; x[d] = rotate(x[d] ^ x[a], 8)
        x[c] += x[d]; x[b] = rotate(x[b] ^ x[c], 7)

    for _ in range(6):
        for column in [(0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15)]:
            quarter(*column)
        for diagonal in [(0, 5, 10, 15), (1, 6, 11, 12), (2, 7, 8, 13), (3, 4, 9, 14)]:
            quarter(*diagonal)
    return (x + state).T.reshape(rows, N)


def times(matrix, vector):
    return (matrix * vector).sum(axis=1, dtype=np.uint32)


def encrypt(A, log_p, entries, rng):
    """A fresh secret for every query. Reusing one would reveal the difference of two queries."""
    secret = rng.integers(0, 2**32, size=N, dtype=np.uint32)
    noise = np.rint(rng.normal(0, SIGMA, size=len(A))).astype(np.int64).astype(np.uint32)
    ciphertext = times(A, secret) + noise
    for index, value in entries:
        ciphertext[index] = (int(ciphertext[index]) + (int(value) << (32 - log_p))) % 2**32
    return secret, ciphertext


def decrypt(answer, hint, secret, log_p):
    shift = 32 - log_p
    return (answer - times(hint, secret) + np.uint32(1 << (shift - 1))) >> np.uint32(shift)


def varint(data, at):
    value, shift = 0, 0
    while True:
        byte = data[at]
        at += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return value, at


def read_page(page):
    """Two bytes of length, a raw DEFLATE stream, then zero padding."""
    length = int.from_bytes(page[:2], "little")
    raw = zlib.decompress(page[2 : 2 + length], -15)
    documents, at = [], 0
    while at < len(raw):
        fields = []
        for _ in range(3):
            size, at = varint(raw, at)
            fields.append(raw[at : at + size].decode())
            at += size
        documents.append(dict(zip(["title", "text", "url"], fields)))
    return documents


def main():
    parser = argparse.ArgumentParser(description="A private search client in plain Python.")
    parser.add_argument("query", nargs="+")
    parser.add_argument("--server", default="http://127.0.0.1:8377")
    parser.add_argument("--unsaid", default="unsaid", help="path to the unsaid binary")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    query = " ".join(args.query)
    base = args.server.rstrip("/")
    rng = np.random.default_rng()

    # Step 1: the public files. A real client caches these.
    manifest = json.loads(get(base, "/v1/manifest"))
    assert manifest["format"] == "unsaid-index-1"
    assert manifest["lwe"] == {"n": N, "log_q": 32, "sigma": SIGMA}, "refuse other parameters"
    dim, model_dim, count = manifest["dim"], manifest["model_dim"], manifest["prototypes"]
    routing = blob(base, manifest, "routing")
    floats = np.frombuffer(routing, dtype="<f4", count=model_dim + dim * model_dim + count * dim)
    mean = floats[:model_dim]
    projection = floats[model_dim : model_dim + dim * model_dim].reshape(dim, model_dim)
    prototypes = floats[model_dim + dim * model_dim :].reshape(count, dim)
    owners = np.frombuffer(routing, dtype="<u4", offset=floats.nbytes)
    layout = np.frombuffer(blob(base, manifest, "layout"), dtype="<u4")
    sizes = layout[: manifest["clusters"]]
    page_first = layout[manifest["clusters"] :]
    digests = blob(base, manifest, "page_digests")
    rank_rows, rank_cols = manifest["rank"]["rows"], manifest["rank"]["cols"]
    page_rows, page_cols = manifest["payload"]["rows"], manifest["payload"]["cols"]
    rank_hint = np.frombuffer(blob(base, manifest, "rank_hint"), dtype="<u4").reshape(rank_rows, N)
    page_hint = np.frombuffer(blob(base, manifest, "payload_hint"), dtype="<u4").reshape(page_rows, N)
    print(f"{manifest['name']}: {manifest['documents']} documents")

    print("expanding the public matrices")
    rank_A = public_matrix(matrix_seed(manifest, "rank"), rank_cols)
    page_A = public_matrix(matrix_seed(manifest, "payload"), page_cols)

    # Step 2: turn the question into a vector, on this machine.
    output = subprocess.run([args.unsaid, "embed", query], capture_output=True, check=True)
    vector = np.array(json.loads(output.stdout), dtype=np.float32)
    projected = projection @ (vector - mean)
    projected /= np.linalg.norm(projected)

    # Step 3: choose clusters using only public data. The server never learns this choice.
    affinity = prototypes @ projected
    best = np.full(manifest["clusters"], -np.inf)
    np.maximum.at(best, owners, affinity)
    clusters = np.argsort(-best, kind="stable")[: manifest["probes"]]

    # Step 4: one encrypted ranking query per cluster.
    log_p = manifest["rank"]["log_p"]
    plain = np.rint(projected * manifest["query_scale"]).astype(np.int64)
    secrets, parts = [], []
    for cluster in clusters:
        entries = [(int(cluster) * dim + i, v) for i, v in enumerate(plain)]
        secret, ciphertext = encrypt(rank_A, log_p, entries, rng)
        secrets.append(secret)
        parts.append(ciphertext.astype("<u4").tobytes())
    answer = post(base, "/v1/rank", b"".join(parts))
    answers = np.frombuffer(answer, dtype="<u4").reshape(len(clusters), rank_rows)

    # Step 5: decrypt the scores. A row beyond a cluster's size is padding.
    starts = np.concatenate([[0], np.cumsum(sizes, dtype=np.int64)])
    ranked = []
    for cluster, secret, part in zip(clusters, secrets, answers):
        values = decrypt(part, rank_hint, secret, log_p).astype(np.int64)
        values = (values << (64 - log_p)) >> (64 - log_p)
        for row in range(int(sizes[cluster])):
            ranked.append((int(values[row]), int(starts[cluster]) + row))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    # Step 6: take the pages that hold the best documents, up to the agreed number.
    def page_of(doc):
        return int(np.searchsorted(page_first, doc, side="right")) - 1

    pages = []
    for _, doc in ranked:
        if page_of(doc) not in pages:
            pages.append(page_of(doc))
        if len(pages) == manifest["pages"]:
            break

    # Step 7: fetch them, one encrypted unit vector per page.
    secrets, parts = [], []
    for page in pages:
        secret, ciphertext = encrypt(page_A, 8, [(page, 1)], rng)
        secrets.append(secret)
        parts.append(ciphertext.astype("<u4").tobytes())
    answer = post(base, "/v1/fetch", b"".join(parts))
    answers = np.frombuffer(answer, dtype="<u4").reshape(len(pages), page_rows)
    print(f"sent {sum(map(len, parts)) + rank_cols * 4 * len(clusters)} bytes of ciphertext\n")

    # Step 8: decrypt, check against the published digests, and show the results.
    found = {}
    for page, secret, part in zip(pages, secrets, answers):
        data = decrypt(part, page_hint, secret, 8).astype(np.uint8).tobytes()
        digest = hashlib.sha256(data).digest()[:DIGEST_BYTES]
        assert digest == digests[page * DIGEST_BYTES : (page + 1) * DIGEST_BYTES], "corrupt page"
        for offset, document in enumerate(read_page(data)):
            found[int(page_first[page]) + offset] = document
    scale = manifest["doc_scale"] * manifest["query_scale"]
    shown = 0
    for score, doc in ranked:
        if doc in found:
            print(f"{score / scale:.2f}  {found[doc]['title']}\n      {found[doc]['text'][:160]}\n")
            shown += 1
            if shown == args.k:
                break


if __name__ == "__main__":
    main()
