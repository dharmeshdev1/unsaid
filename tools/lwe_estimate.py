#!/usr/bin/env python3
"""Estimate the hardness of the LWE parameters under the Core-SVP cost model.

Core-SVP counts one call to an SVP oracle in dimension beta, at 2^(0.292 beta) classically and
2^(0.265 beta) with quantum speedups. It ignores memory and every polynomial factor, so it is a
deliberately pessimistic lower bound that is useful for comparing schemes rather than as a claim
about real attack cost. It is the model the Kyber (ML-KEM) specification tabulates, so this script
first reproduces those figures and then applies the same formulas here.

    python3 tools/lwe_estimate.py

For a finer estimate use https://github.com/malb/lattice-estimator
"""
from math import e, exp, log, log2, pi, sqrt

CLASSICAL, QUANTUM = 0.292, 0.265
SIEVE_OUTPUT = 0.2075


def delta(beta):
    """Root Hermite factor reached by BKZ with block size beta."""
    return ((pi * beta) ** (1 / beta) * beta / (2 * pi * e)) ** (1 / (2 * (beta - 1)))


def primal(n, log_q, sigma, sigma_s=None, max_samples=None):
    """Smallest block size for the primal uSVP attack (the 2016 estimate of Alkim et al.).

    The attacker first moves to normal form, so the secret is as small as the error. A uniform
    secret therefore gets no credit for being uniform, and the attacker spends n samples on this.
    """
    sigma_s = sigma if sigma_s is None else sigma_s
    samples = max_samples if max_samples else 8 * n
    for beta in range(60, 2 * n):
        log_delta = log2(delta(beta))
        for m in range(max(1, n // 4), samples + 1, 8):
            d = m + n + 1
            # Rescale the secret when it is smaller than the error, as the Kyber analysis does.
            scale = sigma / sigma_s
            log_volume = m * log_q + n * log2(scale)
            target = log2(sigma) + 0.5 * log2(beta)
            if target <= (2 * beta - d - 1) * log_delta + log_volume / d:
                return beta
    raise ValueError("no block size found")


def dual(n, log_q, sigma, sigma_s=None, max_samples=None):
    """Lowest classical cost of the dual distinguishing attack, returned as a block size.

    One BKZ run with a sieve yields about 2^(0.2075 beta) short vectors, and distinguishing with
    advantage eps needs about 1/eps^2 of them. Repeat runs are charged when one is not enough.
    """
    sigma_s = sigma if sigma_s is None else sigma_s
    samples = max_samples if max_samples else 8 * n
    best, best_beta = float("inf"), None
    for beta in range(60, 2 * n, 2):
        log_delta = log2(delta(beta))
        for m in range(max(1, n // 4), samples + 1, 8):
            d = m + n
            scale = sigma / sigma_s
            log_length = (d - 1) * log_delta + (n * log_q - n * log2(scale)) / d
            # Standard deviation of the inner product, as a fraction of q.
            tau = 2 ** (log_length + log2(sigma) - log_q)
            exponent = 2 * pi * pi * tau * tau
            log_eps = -exponent / log(2)
            repeats = max(0.0, -2 * log_eps - SIEVE_OUTPUT * beta)
            cost = CLASSICAL * beta + repeats
            if cost < best:
                best, best_beta = cost, cost / CLASSICAL
    return best_beta


def row(name, *args, **kwargs):
    p = primal(*args, **kwargs)
    d = dual(*args, **kwargs)
    print(
        f"{name:<28} primal beta {p:4d} = 2^{CLASSICAL * p:5.1f} classical, 2^{QUANTUM * p:5.1f} quantum"
        f"   dual 2^{CLASSICAL * d:5.1f} classical"
    )
    return p, d


def main():
    print("Calibration against the ML-KEM (Kyber) round 3 specification.")
    print("Published classical Core-SVP: 512: 2^118, 768: 2^182, 1024: 2^256.\n")
    q = log2(3329)
    row("ML-KEM-512", 512, q, sqrt(1.5), max_samples=512 + 256)
    row("ML-KEM-768", 768, q, 1.0, max_samples=768 + 256)
    row("ML-KEM-1024", 1024, q, 1.0, max_samples=1024 + 256)
    print("\nPrivate retrieval parameters, q = 2^32, sigma = 6.4, samples unlimited.\n")
    row("n = 1024 (common in PIR)", 1024, 32, 6.4)
    row("n = 1280 (Unsaid)", 1280, 32, 6.4)
    row("n = 1536", 1536, 32, 6.4)


if __name__ == "__main__":
    main()
