"""Trimmed mean."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def trimmed_mean(
    x,
    *,
    proportion: float = 0.1,
) -> ESRes:
    """Compute the trimmed mean, discarding a fraction from each tail.

    Parameters
    ----------
    x : array-like
        Observations.
    proportion : float
        Fraction to trim from each end (default 0.1 = 10 %).

    Returns
    -------
    ESRes
    """
    from scipy.stats import trim_mean

    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 1:
        raise ValueError("Need at least 1 finite observation.")
    if not 0.0 <= proportion < 0.5:
        raise ValueError("proportion must be in [0, 0.5).")
    val = trim_mean(a, proportion)
    return ESRes(
        measure="trimmed_mean",
        estimate=float(val),
        n=len(a),
        extra={"proportion": proportion},
    )


tmean = trimmed_mean


def cheatsheet() -> str:
    return "trimmed_mean(x, proportion=0.1) -> Trimmed mean."
