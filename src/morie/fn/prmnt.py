# morie.fn — function file (hadesllm/morie)
"""Permutation entropy."""

import math
from collections import Counter

import numpy as np

from ._containers import ESRes


def permutation_entropy(x, order: int = 3, delay: int = 1, **kwargs) -> ESRes:
    """
    Compute permutation entropy of a time series.

    Maps windows of length *order* to ordinal patterns and computes
    Shannon entropy of the pattern distribution.

    :param x: 1-D array-like time series.
    :param order: Embedding dimension / permutation order (default 3).
    :param delay: Time delay (default 1).
    :return: ESRes with permutation entropy (normalised to [0,1]).

    References
    ----------
    Bandt C, Pompe B (2002). Permutation entropy: a natural complexity
    measure for time series. Physical Review Letters, 88(17), 174102.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < order * delay:
        raise ValueError(f"Need at least {order * delay} observations.")

    patterns: list[tuple[int, ...]] = []
    for i in range(n - (order - 1) * delay):
        window = x[i : i + order * delay : delay]
        patterns.append(tuple(int(j) for j in np.argsort(window)))

    counts = Counter(patterns)
    total = sum(counts.values())
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * math.log2(p)

    h_max = math.log2(math.factorial(order))
    h_norm = h / h_max if h_max > 0 else 0.0

    return ESRes(
        measure="permutation_entropy",
        estimate=h_norm,
        n=n,
        extra={"raw_entropy": h, "max_entropy": h_max, "order": order, "delay": delay},
    )


prmnt = permutation_entropy


def cheatsheet() -> str:
    return "permutation_entropy(x, order=3) -> Normalised permutation entropy."
