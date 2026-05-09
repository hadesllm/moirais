"""Substance use prevalence with Wilson confidence interval."""

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def substance_prevalence(
    n_users: int,
    n_surveyed: int,
    weights: np.ndarray | None = None,
    confidence: float = 0.95,
) -> ESRes:
    """Estimate substance use prevalence with confidence interval.

    .. math::

        \\hat{p} = \\frac{n_{\\text{users}}}{n_{\\text{surveyed}}}

    If *weights* are provided, uses weighted proportion instead.

    Parameters
    ----------
    n_users : int
        Number of substance users in sample.
    n_surveyed : int
        Total number surveyed (must be > 0).
    weights : array-like or None
        Optional survey weights (length *n_surveyed*).
    confidence : float, default 0.95

    Returns
    -------
    ESRes
    """
    if n_surveyed <= 0:
        raise ValueError("n_surveyed must be positive")
    if n_users < 0 or n_users > n_surveyed:
        raise ValueError("n_users must be in [0, n_surveyed]")

    if weights is not None:
        w = np.asarray(weights, dtype=float)
        p = np.sum(w[:n_users]) / np.sum(w) if len(w) == n_surveyed else n_users / n_surveyed
    else:
        p = n_users / n_surveyed

    z = stats.norm.ppf((1 + confidence) / 2)
    denom = 1 + z**2 / n_surveyed
    centre = p + z**2 / (2 * n_surveyed)
    margin = z * np.sqrt(p * (1 - p) / n_surveyed + z**2 / (4 * n_surveyed**2))
    ci_lo = (centre - margin) / denom
    ci_hi = (centre + margin) / denom

    return ESRes(
        measure="substance_prevalence",
        estimate=float(p),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        n=n_surveyed,
    )


suprv = substance_prevalence


def cheatsheet() -> str:
    return "substance_prevalence({}) -> Substance use prevalence with Wilson confidence interval."
