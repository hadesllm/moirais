"""Winsorized mean."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def winsorized_mean(
    x,
    *,
    proportion: float = 0.1,
) -> ESRes:
    """Compute the Winsorized mean, clamping extreme values.

    Parameters
    ----------
    x : array-like
        Observations.
    proportion : float
        Fraction to Winsorize from each end (default 0.1 = 10 %).

    Returns
    -------
    ESRes
    """
    from scipy.stats.mstats import winsorize as _winsorize

    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 1:
        raise ValueError("Need at least 1 finite observation.")
    if not 0.0 <= proportion < 0.5:
        raise ValueError("proportion must be in [0, 0.5).")
    w = _winsorize(a, limits=(proportion, proportion))
    val = float(np.mean(w))
    return ESRes(
        measure="winsorized_mean",
        estimate=val,
        n=len(a),
        extra={"proportion": proportion},
    )


wmean = winsorized_mean


def cheatsheet() -> str:
    return "winsorized_mean(x, proportion=0.1) -> Winsorized mean."
