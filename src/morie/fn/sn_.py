"""Rousseeuw-Croux Sn scale estimator."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def sn_estimator(x: np.ndarray) -> ESRes:
    r"""Sn scale estimator (Rousseeuw & Croux, 1993).

    Sn = c_n * 1.1926 * med_i { med_j |x_i - x_j| }

    where c_n is a finite-sample correction.

    Parameters
    ----------
    x : array-like

    Returns
    -------
    ESRes
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        raise ValueError("Need >= 2 finite observations.")

    inner_medians = np.empty(n)
    for i in range(n):
        inner_medians[i] = np.median(np.abs(x[i] - x))
    raw = float(np.median(inner_medians))

    if n <= 9:
        cn = [0, 0, 0.743, 1.851, 0.954, 1.351, 0.993, 1.198, 0.994, 1.131]
        correction = cn[n] if n < len(cn) else 1.0
    else:
        correction = n / (n - 0.9)

    sn = 1.1926 * correction * raw

    return ESRes(
        measure="sn",
        estimate=float(sn),
        n=n,
        extra={"raw_median": float(raw), "correction": correction},
    )


sn_ = sn_estimator


def cheatsheet() -> str:
    return "sn_estimator({}) -> Rousseeuw-Croux Sn scale estimator."
