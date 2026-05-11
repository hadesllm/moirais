# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Aberration detection (Farrington algorithm)."""

from __future__ import annotations

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def farrington_aberration(
    counts: list[int] | np.ndarray,
    baselines: list[float] | np.ndarray | None = None,
    alpha: float = 0.05,
    overdispersion: bool = True,
) -> ESRes:
    """Aberration detection using the Farrington algorithm.

    Computes expected counts from a quasi-Poisson model and flags
    observations exceeding the upper prediction limit.

    Parameters
    ----------
    counts : array-like of int
        Observed counts per time period.
    baselines : array-like of float, optional
        Expected baseline counts. If None, uses rolling mean of
        first half of data.
    alpha : float, default 0.05
        Significance level for the upper threshold.
    overdispersion : bool, default True
        Adjust for overdispersion (quasi-Poisson).

    Returns
    -------
    ESRes
        estimate is number of aberrations detected.

    References
    ----------
    Farrington, C. P. et al. (1996). A statistical algorithm for the
    early detection of outbreaks of infectious disease. Journal of
    the Royal Statistical Society A, 159(3), 547-563.
    """
    c = np.asarray(counts, dtype=float)
    n = len(c)
    if n < 4:
        raise ValueError("Need at least 4 time periods")

    if baselines is None:
        half = n // 2
        mu = np.full(n, np.mean(c[:half]))
    else:
        mu = np.asarray(baselines, dtype=float)

    if overdispersion:
        residuals = c[:len(c) // 2] - mu[:len(c) // 2]
        phi = max(1.0, float(np.sum(residuals**2 / np.maximum(mu[:len(c) // 2], 0.5)) / max(len(residuals) - 1, 1)))
    else:
        phi = 1.0

    z = stats.norm.ppf(1 - alpha)
    thresholds = mu + z * np.sqrt(phi * np.maximum(mu, 0.5))

    aberrations = []
    for t in range(n):
        if c[t] > thresholds[t]:
            aberrations.append(t)

    return ESRes(
        measure="farrington",
        estimate=float(len(aberrations)),
        n=n,
        extra={
            "aberrations": aberrations,
            "thresholds": thresholds.tolist(),
            "expected": mu.tolist(),
            "phi": float(phi),
            "alpha": alpha,
        },
    )


abrtm = farrington_aberration


def cheatsheet() -> str:
    return "farrington_aberration({}) -> Farrington aberration detection algorithm."
