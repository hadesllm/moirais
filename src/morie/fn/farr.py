# morie.fn -- function file (hadesllm/morie)
"""Farrington algorithm for aberration detection."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import DescriptiveResult


def farrington_detect(
    current: float,
    historical: np.ndarray | list,
    *,
    alpha: float = 0.01,
    w: int = 3,
) -> DescriptiveResult:
    """
    Simplified Farrington algorithm for aberration detection.

    Compares current count to a quasi-Poisson model fitted to
    historical baselines.

    Parameters
    ----------
    current : float
        Current period count.
    historical : array-like
        Historical baseline counts (same period in prior years).
    alpha : float
        Significance level for exceedance.
    w : int
        Window width around reference period.

    Returns
    -------
    DescriptiveResult
        extra has 'threshold', 'exceedance', 'expected', 'overdispersion'.

    References
    ----------
    Farrington, C. P., et al. (1996). A statistical algorithm for the
    early detection of outbreaks of infectious disease. *J R Stat Soc
    Ser A*, 159(3), 547-563.
    """
    hist = np.asarray(historical, dtype=float)
    if len(hist) < 3:
        raise ValueError("Need at least 3 historical values.")

    mu = float(np.mean(hist))
    var = float(np.var(hist, ddof=1))
    phi = max(var / mu, 1.0) if mu > 0 else 1.0

    se = np.sqrt(phi * mu / len(hist))
    z = stats.norm.ppf(1 - alpha)
    threshold = mu + z * np.sqrt(phi * mu)
    exceedance = current > threshold

    return DescriptiveResult(
        name="Farrington",
        value=float(current),
        extra={
            "threshold": float(threshold),
            "exceedance": bool(exceedance),
            "expected": mu,
            "overdispersion": phi,
            "z": float(z),
        },
    )


farr = farrington_detect


def cheatsheet() -> str:
    return "farrington_detect({}) -> Farrington algorithm for aberration detection."
