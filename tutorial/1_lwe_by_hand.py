#!/usr/bin/env python3
"""Learning with errors, with numbers small enough to check by hand.

Run it: python3 tutorial/1_lwe_by_hand.py

The only library is numpy. By the end you will have seen the single trick the whole search engine
rests on: a server can multiply its data by an encrypted vector without being able to read it.
"""
import numpy as np

rng = np.random.default_rng(7)

print("Part 1. Equations are easy to solve, until they are slightly wrong.\n")

q = 97
secret = np.array([12, 45, 3])
A = rng.integers(0, q, size=(3, 3))
exact = A @ secret % q


def solve_mod(A, b, q):
    """Gaussian elimination modulo a prime, which a first course in linear algebra covers."""
    system = np.column_stack([A, b]) % q
    n = len(A)
    for i in range(n):
        pivot = next(r for r in range(i, n) if system[r, i] % q)
        system[[i, pivot]] = system[[pivot, i]]
        system[i] = system[i] * pow(int(system[i, i]), -1, q) % q
        for r in range(n):
            if r != i:
                system[r] = (system[r] - system[r, i] * system[i]) % q
    return system[:, -1]


print("secret             ", secret)
print("solved from A, b   ", solve_mod(A, exact, q), " exact, found at once")
noisy = (exact + np.array([1, -1, 1])) % q
print("solved with noise  ", solve_mod(A, noisy, q), " noise of 1 ruins everything")
print(
    """
Elimination multiplies errors as it goes, so an error of 1 turns into an answer that has nothing to
do with the secret. With a few hundred unknowns, no known algorithm can recover the secret, even
on a quantum computer. That problem is called learning with errors, and it is the same assumption
behind ML-KEM, the post-quantum standard your browser may already be using.
"""
)

print("Part 2. Hide a message under that noise.\n")

q = 2**32
n = 16
p = 2**8
delta = q // p
A = rng.integers(0, q, size=(32, n), dtype=np.uint32)
secret = rng.integers(0, q, size=n, dtype=np.uint32)
noise = np.rint(rng.normal(0, 6.4, size=32)).astype(np.int64).astype(np.uint32)
message = np.zeros(32, dtype=np.uint32)
message[5] = 42


def times(matrix, vector):
    """Matrix times vector, wrapping modulo 2^32. numpy's uint32 does this by itself."""
    return (matrix * vector).sum(axis=1, dtype=np.uint32)


ciphertext = times(A, secret) + noise + delta * message
print("message   ", message[:8])
print("ciphertext", ciphertext[:4], "...")
print("The numbers are uniformly random to anyone without the secret.")

opened = ciphertext - times(A, secret)
rounded = ((opened + delta // 2) >> 24).astype(np.uint32)
print("decrypted ", rounded[:8], " rounding removes the noise\n")

print("Part 3. The trick: a server computes on it without reading it.\n")

# The server holds a table, and it is not encrypted. The client wants one column of it.
table = rng.integers(0, 256, size=(4, 32), dtype=np.uint32)
print("server's table, column 5:", table[:, 5])

# The server sends this ahead of time. It is M * A, it is public, and it contains no secrets.
hint = np.array([[(row * A[:, j]).sum(dtype=np.uint32) for j in range(n)] for row in table])

# The client encrypts a vector that is all zeros except for a 1 at the column it wants. It uses a
# new secret, because using one secret twice would reveal the difference between two messages.
secret = rng.integers(0, q, size=n, dtype=np.uint32)
noise = np.rint(rng.normal(0, 6.4, size=32)).astype(np.int64).astype(np.uint32)
wanted = np.zeros(32, dtype=np.uint32)
wanted[5] = 1
ciphertext = times(A, secret) + noise + delta * wanted

# The server multiplies its table by the ciphertext. It never sees the message or the secret.
answer = times(table, ciphertext)

# M(As + e + delta x) = (MA)s + Me + delta Mx. The client knows MA and s, so it can remove them.
remainder = answer - times(hint, secret)
recovered = ((remainder + delta // 2) >> 24).astype(np.uint32)
print("client recovered        :", recovered)
print(
    """
The server multiplied its table by what looked like random numbers and sent the result back. It
has no idea that the client asked for column 5. Because it reads the whole table every time, it
could not tell even if it watched its own memory.

Next: python3 tutorial/2_pir_from_scratch.py
"""
)
