"""Sn scale estimator (Rousseeuw & Croux, 1993)."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def sn_estimator(
    x,
    *,
    consistency: float = 1.1926,
) -> ESRes:
    """Rousseeuw-Croux Sn scale estimator.

    Sn = c_n * med_i { med_j |x_i - x_j| }.  High breakdown (50 %)
    and bounded influence.

    Parameters
    ----------
    x : array-like
        Observations.
    consistency : float
        Finite-sample consistency factor (default 1.1926 for normal).

    Returns
    -------
    ESRes
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n < 2:
        raise ValueError("Need at least 2 finite observations.")

    inner_medians = np.array([np.median(np.abs(a[i] - a)) for i in range(n)])
    sn = consistency * float(np.median(inner_medians))

    return ESRes(
        measure="sn_estimator",
        estimate=sn,
        n=n,
        extra={"consistency": consistency},
    )


sn_es = sn_estimator


def cheatsheet() -> str:
    return "sn_estimator(x) -> Rousseeuw-Croux Sn scale estimator."
