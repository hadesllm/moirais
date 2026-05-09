"""Victimization survey rate. 'Stay on target.' -- Gold Five"""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import CrimeResult


def victimization_rate(
    n_victims: int,
    n_surveyed: int,
    weights: np.ndarray | None = None,
    confidence: float = 0.95,
    per: int = 1_000,
) -> CrimeResult:
    """Victimization rate per *per* surveyed, optionally weighted.

    Parameters
    ----------
    n_victims : int
        Count of victimised respondents (used when *weights* is None).
    n_surveyed : int
        Total respondents.
    weights : ndarray or None
        Optional survey weights (length *n_surveyed*).  When supplied the
        weighted proportion replaces the simple ratio.
    confidence : float, default 0.95
    per : int, default 1000

    Returns
    -------
    CrimeResult
    """
    if n_surveyed <= 0:
        raise ValueError("n_surveyed must be positive")
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != n_surveyed:
            raise ValueError("weights length must equal n_surveyed")
        p_hat = float(np.sum(weights[:n_victims]) / np.sum(weights))
    else:
        p_hat = n_victims / n_surveyed
    z = _st.norm.ppf(1 - (1 - confidence) / 2)
    se = np.sqrt(p_hat * (1 - p_hat) / n_surveyed)
    ci_lo = max(0, p_hat - z * se) * per
    ci_hi = min(1, p_hat + z * se) * per
    return CrimeResult(
        name="Victimization rate",
        rate=float(p_hat * per),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        n=n_victims,
        population=n_surveyed,
        extra={"per": per, "confidence": confidence, "weighted": weights is not None},
    )


vctm = victimization_rate


def cheatsheet() -> str:
    return "victimization_rate({}) -> Victimization survey rate. 'Stay on target.' -- Gold Five"
