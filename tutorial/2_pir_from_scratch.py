#!/usr/bin/env python3
"""Private lookup and private ranking from scratch, in one file with no network.

Run it: python3 tutorial/2_pir_from_scratch.py

The server and the client are both here, separated by comments, so you can see that nothing passes
between them except `hint`, `query` and `answer`. These are the two operations a real search needs:
fetch a document, and score many documents against a question.
"""
import numpy as np

rng = np.random.default_rng()

Q_BITS = 32
N = 1280  # The real lattice dimension. See docs/security.md for why.
SIGMA = 6.4


def times(matrix, vector):
    """Matrix times vector modulo 2^32."""
    return (matrix * vector).sum(axis=1, dtype=np.uint32)


def fresh_mask(A):
    """A new secret and the encryption of zero under it."""
    secret = rng.integers(0, 2**32, size=N, dtype=np.uint32)
    noise = np.rint(rng.normal(0, SIGMA, size=len(A))).astype(np.int64).astype(np.uint32)
    return secret, times(A, secret) + noise


def decrypt(answer, hint, secret, log_p):
    """Remove the mask and round the noise away. Values come back in 0..p."""
    shift = Q_BITS - log_p
    half = np.uint32(1 << (shift - 1))
    return (answer - times(hint, secret) + half) >> shift


print("One: fetch a record without saying which.\n")

records = [
    "Mount Everest is 8,849 metres tall.",
    "The Pacific is the largest ocean.",
    "Honey does not spoil if it is sealed.",
    "Octopuses have three hearts.",
    "Light takes eight minutes to arrive from the Sun.",
]
ROW_BYTES = 64

# SERVER: every record becomes one column of a matrix of bytes.
M = np.zeros((ROW_BYTES, len(records)), dtype=np.uint32)
for j, record in enumerate(records):
    data = record.encode()
    M[: len(data), j] = list(data)

# PUBLIC: anyone can derive A from a seed. The real system uses ChaCha12, so nobody picks it.
A = rng.integers(0, 2**32, size=(len(records), N), dtype=np.uint32)

# SERVER: the hint is computed once and downloaded by every client. It reveals nothing about any
# query, because no query exists yet.
hint = M @ A  # numpy wraps uint32 multiplication, so this is M * A modulo 2^32.

# CLIENT: encrypt the unit vector that selects record 3.
wanted = 3
secret, query = fresh_mask(A)
query[wanted] += np.uint32(1 << (Q_BITS - 8))
print("the server receives:", " ".join(f"{v:08x}" for v in query))

# SERVER: the whole of its work.
answer = times(M, query)

# CLIENT:
data = decrypt(answer, hint, secret, log_p=8).astype(np.uint8).tobytes()
print("the client reads   :", data.rstrip(b"\0").decode(), "\n")

print("Two: score documents against a question without saying what it is.\n")

# In the real system, a neural network turns text into vectors like these. Here we make up three
# meanings so that you can check the result by eye: space, cooking, and money.
documents = {
    "How rockets reach orbit": [0.9, 0.1, 0.0],
    "Life on the space station": [0.8, 0.2, 0.1],
    "Baking bread at home": [0.0, 0.9, 0.1],
    "Fermenting vegetables": [0.1, 0.8, 0.0],
    "Paying off credit card debt": [0.0, 0.1, 0.9],
    "What bankruptcy means": [0.1, 0.0, 0.8],
}
SCALE = 32
titles = list(documents)
vectors = np.array(list(documents.values()))
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

# SERVER: scores are a matrix product, so the document vectors are simply the matrix. It stores
# them as small integers so that the noise stays small.
M = np.rint(vectors * SCALE).astype(np.int64).astype(np.uint32)
A = rng.integers(0, 2**32, size=(M.shape[1], N), dtype=np.uint32)
hint = M @ A

# CLIENT: the question is secret. Encrypt it the same way, and use a larger modulus because
# scores are bigger than bytes.
question = np.array([0.1, 0.0, 0.95])
question /= np.linalg.norm(question)
LOG_P = 14
secret, query = fresh_mask(A)
plain = np.rint(question * SCALE).astype(np.int64).astype(np.uint32)
query = query + (plain << np.uint32(Q_BITS - LOG_P))

# SERVER: the same single operation as before. It cannot tell a lookup from a ranking.
answer = times(M, query)

# CLIENT: read the scores as signed numbers and sort them.
scores = decrypt(answer, hint, secret, LOG_P).astype(np.int64)
scores = (scores << (64 - LOG_P)) >> (64 - LOG_P)
for score, title in sorted(zip(scores / SCALE**2, titles), reverse=True):
    print(f"  {score:5.2f}  {title}")

print(
    """
The server did one matrix product per query and learned nothing. Everything else in Unsaid exists
to make this practical at scale: grouping documents into clusters so that a query only has to carry
a few of them, and packing documents into pages so that the top results can be fetched the same way.

Next: read docs/how-it-works.md, then talk to a real server with tutorial/3_real_client.py
"""
)
