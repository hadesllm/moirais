# morie.fn — function file (hadesllm/morie)
"""Sieve of Eratosthenes for prime generation."""

import numpy as np

from ._containers import DescriptiveResult
def prime_sieve(n: int, **kwargs) -> DescriptiveResult:
    r"""
    Generate all primes up to *n* using the Sieve of Eratosthenes.

    Time complexity :math:`O(n \\log \\log n)`, space :math:`O(n)`.

    The Prime Number Theorem gives :math:`\\pi(n) \\sim n / \\ln n`.

    :param n: Upper bound (inclusive). Must be >= 2.
    :return: DescriptiveResult with prime count and primes array.
    :raises ValueError: If n < 2.

    References
    ----------
    Hardy, G. H. & Wright, E. M. (2008). *An Introduction to the Theory
    of Numbers* (6th ed.). Oxford University Press.
    """
    n = int(n)
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}.")

    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if sieve[i]:
            sieve[i * i :: i] = False

    primes = np.nonzero(sieve)[0]
    count = len(primes)

    return DescriptiveResult(
        name="prime_sieve",
        value=float(count),
        extra={
            "prime_count": count,
            "primes": primes,
            "n": n,
            "density": count / n if n > 0 else 0.0,
            "largest_prime": int(primes[-1]) if count > 0 else None,
        },
    )


prsiv = prime_sieve


def cheatsheet() -> str:
    return "prime_sieve({}) -> Sieve of Eratosthenes for prime generation."
