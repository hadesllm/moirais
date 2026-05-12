# morie.fn -- function file (hadesllm/morie)
"""Standard Error of Measurement."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn.crba import crba


def rsem(
    data: pd.DataFrame | np.ndarray,
    *,
    reliability: float | None = None,
) -> float:
    """Standard Error of Measurement (SEM).

    SEM = SD_total * sqrt(1 - reliability)

    The SEM quantifies the precision of individual scores.  A smaller
    SEM indicates more precise measurement.

    Parameters
    ----------
    data : DataFrame or ndarray
        Item matrix (respondents x items).
    reliability : float or None
        Reliability estimate to use.  If None (default), Cronbach's
        alpha is computed from ``data`` via ``crba()``.

    Returns
    -------
    float
        Standard Error of Measurement.

    References
    ----------
    Harvill, L. M. (1991). Standard error of measurement.
    *Educational Measurement: Issues and Practice*, 10(2), 33-41.
    """
    X = np.asarray(data, dtype=np.float64)
    mask = np.all(np.isfinite(X), axis=1)
    X = X[mask]
    n, k = X.shape

    if k < 2 or n < 2:
        return float("nan")

    if reliability is None:
        res = crba(data)
        reliability = res.raw

    if np.isnan(reliability):
        return float("nan")

    # Clamp reliability to [0, 1) to avoid sqrt of negative
    rel = float(np.clip(reliability, 0.0, 1.0))

    total = X.sum(axis=1)
    sd_total = float(np.std(total, ddof=1))

    if rel >= 1.0:
        return 0.0

    sem = sd_total * np.sqrt(1.0 - rel)
    return float(sem)


short = rsem


def cheatsheet() -> str:
    return "rsem({}) -> Standard Error of Measurement."
