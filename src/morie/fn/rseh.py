# morie.fn — function file (hadesllm/morie)
"""Standard Error of Measurement with confidence interval."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp

from morie.fn.crba import crba
from morie.fn.rsem import rsem


def rseh(
    data: pd.DataFrame | np.ndarray,
    *,
    reliability: float | None = None,
    alpha: float = 0.05,
) -> dict:
    """SEM with confidence band for observed scores.

    Returns the SEM and the half-width of the confidence interval
    around any observed score:  X_true = X_observed +/- z * SEM.

    Parameters
    ----------
    data : DataFrame or ndarray
        Item matrix (respondents x items).
    reliability : float or None
        Reliability estimate.  If None, uses Cronbach's alpha.
    alpha : float
        Significance level for the CI (default 0.05 for 95% CI).

    Returns
    -------
    dict
        Keys: ``sem`` (float), ``ci_width`` (float, half-width),
        ``reliability`` (float), ``confidence`` (float).

    References
    ----------
    Harvill, L. M. (1991). Standard error of measurement.
    *Educational Measurement: Issues and Practice*, 10(2), 33-41.
    """
    X = np.asarray(data, dtype=np.float64)
    mask = np.all(np.isfinite(X), axis=1)
    X = X[mask]

    if reliability is None:
        res = crba(data)
        reliability = res.raw

    sem_val = rsem(data, reliability=reliability)

    z = sp.norm.ppf(1.0 - alpha / 2.0)
    ci_width = z * sem_val

    return {
        "sem": float(sem_val),
        "ci_width": float(ci_width),
        "reliability": float(reliability),
        "confidence": float(1.0 - alpha),
    }


short = rseh


def cheatsheet() -> str:
    return "rseh({}) -> Standard Error of Measurement with confidence interval."
