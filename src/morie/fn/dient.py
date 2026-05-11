# morie.fn — function file (hadesllm/morie)
"""Dispersion entropy."""

import math

import numpy as np
from scipy.special import erfc  # noqa: F401
from scipy.stats import norm

from ._containers import ESRes


def dispersion_entropy(x, m: int = 2, c: int = 6, delay: int = 1, cdf=None, **kwargs) -> ESRes:
    """
    Compute dispersion entropy (DispEn) of a time series.

    Maps values to c classes via normal CDF, then computes Shannon
    entropy over dispersion patterns.

    :param x: 1-D array-like time series.
    :param m: Embedding dimension (default 2).
    :param c: Number of classes (default 6).
    :param delay: Time delay (default 1).
    :return: ESRes with normalised dispersion entropy in [0,1].

    References
    ----------
    Rostaghi M, Azami H (2016). Dispersion entropy: A measure for
    time-series analysis. IEEE Signal Processing Letters, 23(5), 610-614.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < m * delay + 1:
        raise ValueError(f"Need at least {m * delay + 1} observations.")

    mu, sigma = np.mean(x), np.std(x, ddof=1)
    if sigma == 0:
        raise ValueError("Time series has zero variance.")
    z = norm.cdf(x, loc=mu, scale=sigma)
    classes = np.clip(np.ceil(z * c).astype(int), 1, c)

    patterns: dict[tuple, int] = {}
    total = 0
    for i in range(n - (m - 1) * delay):
        pat = tuple(classes[i + j * delay] for j in range(m))
        patterns[pat] = patterns.get(pat, 0) + 1
        total += 1

    h = 0.0
    for cnt in patterns.values():
        p = cnt / total
        if p > 0:
            h -= p * math.log2(p)

    h_max = m * math.log2(c)
    h_norm = h / h_max if h_max > 0 else 0.0

    return ESRes(
        measure="dispersion_entropy",
        estimate=h_norm,
        n=n,
        extra={"raw_entropy": h, "max_entropy": h_max, "m": m, "c": c, "n_patterns": len(patterns)},
    )


dient = dispersion_entropy


def cheatsheet() -> str:
    return "dispersion_entropy(x, m=2, c=6) -> Normalised dispersion entropy."
