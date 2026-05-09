# moirais.fn — function file (hadesllm/moirais)
"""Prime density — pi(n) counting function."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def prime_density(n: int) -> ESRes:
    """Prime counting function pi(n) and density pi(n)/n.

    Uses sieve of Eratosthenes.

    Parameters
    ----------
    n : int (>= 2)

    Returns
    -------
    ESRes
    """
    if n < 2:
        raise ValueError("n must be >= 2.")

    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if sieve[i]:
            sieve[i * i :: i] = False

    pi_n = int(sieve.sum())
    density = pi_n / n

    return ESRes(
        measure="prime_density",
        estimate=float(density),
        n=n,
        extra={"pi_n": pi_n, "pnt_approx": float(n / np.log(n)) if n > 1 else 0.0},
    )


prime = prime_density


def cheatsheet() -> str:
    return "prime_density({}) -> Prime density — pi(n) counting function."
