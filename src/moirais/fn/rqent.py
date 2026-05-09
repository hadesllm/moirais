# moirais.fn — function file (hadesllm/moirais)
"""Recurrence quantification entropy."""

import math

import numpy as np

from ._containers import ESRes


def recurrence_entropy(x, m: int = 2, delay: int = 1, eps: float | None = None, **kwargs) -> ESRes:
    """
    Compute entropy of diagonal line length distribution from
    recurrence quantification analysis (RQA).

    :param x: 1-D array-like time series.
    :param m: Embedding dimension (default 2).
    :param delay: Time delay (default 1).
    :param eps: Recurrence threshold (default 0.1 * std(x)).
    :return: ESRes with recurrence entropy.

    References
    ----------
    Marwan N et al. (2007). Recurrence plots for the analysis of
    complex systems. Physics Reports, 438(5-6), 237-329.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < m * delay + 2:
        raise ValueError(f"Need at least {m * delay + 2} observations.")
    if eps is None:
        eps = 0.1 * np.std(x, ddof=1)
    if eps <= 0:
        raise ValueError("eps must be positive.")

    n_embed = n - (m - 1) * delay
    embedded = np.array([x[i:i + m * delay:delay] for i in range(n_embed)])

    diag_lengths: list[int] = []
    for offset in range(1, n_embed):
        length = 0
        for i in range(n_embed - offset):
            j = i + offset
            dist = np.max(np.abs(embedded[i] - embedded[j]))
            if dist < eps:
                length += 1
            else:
                if length > 0:
                    diag_lengths.append(length)
                length = 0
        if length > 0:
            diag_lengths.append(length)

    if not diag_lengths:
        return ESRes(measure="recurrence_entropy", estimate=0.0, n=n, extra={"n_diag_lines": 0})

    from collections import Counter
    counts = Counter(diag_lengths)
    total = sum(counts.values())
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * math.log2(p)

    return ESRes(
        measure="recurrence_entropy",
        estimate=float(h),
        n=n,
        extra={"n_diag_lines": total, "max_line": max(diag_lengths), "m": m, "eps": eps},
    )


rqent = recurrence_entropy


def cheatsheet() -> str:
    return "recurrence_entropy(x) -> RQA diagonal line entropy."
