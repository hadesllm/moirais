# moirais.fn — function file (hadesllm/moirais)
"""Qn scale estimator (Rousseeuw & Croux, 1993)."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def qn_estimator(
    x,
    *,
    consistency: float = 2.2219,
) -> ESRes:
    """Rousseeuw-Croux Qn scale estimator.

    Qn = c_n * {|x_i - x_j|; i < j}_(k), the k-th order statistic
    of pairwise differences with k = binom(floor(n/2)+1, 2).

    Parameters
    ----------
    x : array-like
        Observations.
    consistency : float
        Finite-sample consistency factor (default 2.2219 for normal).

    Returns
    -------
    ESRes
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n < 2:
        raise ValueError("Need at least 2 finite observations.")

    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(abs(a[i] - a[j]))
    diffs = np.sort(diffs)

    h = n // 2 + 1
    k = h * (h - 1) // 2
    idx = min(k - 1, len(diffs) - 1)
    qn = consistency * float(diffs[idx])

    return ESRes(
        measure="qn_estimator",
        estimate=qn,
        n=n,
        extra={"consistency": consistency, "k": k},
    )


qn_es = qn_estimator


def cheatsheet() -> str:
    return "qn_estimator(x) -> Rousseeuw-Croux Qn scale estimator."
