# morie.fn — function file (hadesllm/morie)
"""Minimal Detectable Change."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp

from morie.fn.rsem import rsem


def rmdc(
    data: pd.DataFrame | np.ndarray | None = None,
    *,
    sem: float | None = None,
    reliability: float | None = None,
    confidence: float = 0.95,
) -> float:
    """Minimal Detectable Change (MDC).

    MDC = z * SEM * sqrt(2)

    The smallest change in score that exceeds measurement error at the
    given confidence level.  Provide either ``data`` (from which SEM is
    computed) or ``sem`` directly.

    Parameters
    ----------
    data : DataFrame, ndarray, or None
        Item matrix.  Required if ``sem`` is not provided.
    sem : float or None
        Pre-computed SEM.  If None, computed from ``data``.
    reliability : float or None
        Reliability for SEM computation (only used when ``sem`` is
        None).  If None, Cronbach's alpha is used.
    confidence : float
        Confidence level (default 0.95).

    Returns
    -------
    float
        Minimal Detectable Change value.

    Raises
    ------
    ValueError
        If neither ``data`` nor ``sem`` is provided.

    References
    ----------
    Beckerman, H., et al. (2001). Smallest real difference, a link
    between reproducibility and responsiveness. *Quality of Life
    Research*, 10(7), 571-578.
    """
    if sem is None:
        if data is None:
            raise ValueError("Must provide either 'data' or 'sem'.")
        sem = rsem(data, reliability=reliability)

    if np.isnan(sem):
        return float("nan")

    z = sp.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    mdc = z * sem * np.sqrt(2.0)
    return float(mdc)


short = rmdc


def cheatsheet() -> str:
    return "rmdc({}) -> Minimal Detectable Change."
