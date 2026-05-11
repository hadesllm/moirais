# morie.fn — function file (hadesllm/morie)
"""Single EMD sifting pass to extract one IMF."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Mastering others is strength; mastering yourself is true power. — Lao Tzu"


def emd_sifting(x, max_iter: int = 300, tol: float = 1e-6, **kwargs) -> DescriptiveResult:
    """Single EMD sifting pass: extract one Intrinsic Mode Function.

    Iteratively removes the local mean envelope until the result
    satisfies IMF conditions (symmetric envelopes, zero-mean).

    Parameters
    ----------
    x : array-like
        1-D input signal.
    max_iter : int
        Maximum sifting iterations (default 300).
    tol : float
        Stopping tolerance on relative envelope energy (default 1e-6).

    Returns
    -------
    DescriptiveResult
        ``value`` is number of iterations used; ``extra`` has ``imf``,
        ``residual``.

    References
    ----------
    Huang, N. E., et al. (1998). The empirical mode decomposition and
    the Hilbert spectrum for nonlinear and non-stationary time series
    analysis. *Proc. R. Soc. Lond. A*, 454, 903-995.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    h = x.copy()
    n_iter = 0
    t = np.arange(N)
    for i in range(max_iter):
        maxima = []
        minima = []
        for j in range(1, N - 1):
            if h[j] > h[j - 1] and h[j] >= h[j + 1]:
                maxima.append(j)
            if h[j] < h[j - 1] and h[j] <= h[j + 1]:
                minima.append(j)
        if len(maxima) < 2 or len(minima) < 2:
            n_iter = i + 1
            break
        upper = np.interp(t, maxima, h[maxima])
        lower = np.interp(t, minima, h[minima])
        mean_env = (upper + lower) / 2.0
        rel_energy = np.sum(mean_env**2) / (np.sum(h**2) + 1e-15)
        h = h - mean_env
        n_iter = i + 1
        if rel_energy < tol:
            break
    residual = x - h
    return DescriptiveResult(
        name="emd_sifting",
        value=n_iter,
        extra={"imf": h, "residual": residual, "max_iter": max_iter},
    )


emdsi = emd_sifting


def cheatsheet() -> str:
    return "emd_sifting({}) -> Single EMD sifting pass to extract one IMF."
