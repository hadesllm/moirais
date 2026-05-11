"""Trimmed mean."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def trimmed_mean(
    x: np.ndarray,
    *,
    trim: float = 0.1,
) -> ESRes:
    """Trimmed mean — discards a fraction from each tail before averaging.

    Parameters
    ----------
    x : array-like
    trim : float
        Fraction to trim from each side (0 to 0.5).

    Returns
    -------
    ESRes
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        raise ValueError("Need >= 3 finite observations.")
    if not 0.0 <= trim < 0.5:
        raise ValueError("trim must be in [0, 0.5).")

    k = int(np.floor(n * trim))
    xs = np.sort(x)
    trimmed = xs[k : n - k] if k > 0 else xs
    tm = float(np.mean(trimmed))
    se = float(np.std(trimmed, ddof=1) / np.sqrt(len(trimmed)))

    return ESRes(
        measure="trimmed_mean",
        estimate=tm,
        se=se,
        n=n,
        extra={"trim": trim, "k_trimmed": k, "n_kept": len(trimmed)},
    )


trimm = trimmed_mean


def cheatsheet() -> str:
    return "trimmed_mean({}) -> Trimmed mean."
