# morie.fn — function file (hadesllm/morie)
"""Exposure distribution analysis."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import DescriptiveResult


def exposure_assessment(
    exposure_levels: np.ndarray | list,
    health_outcomes: np.ndarray | list,
    *,
    method: str = "quartile",
) -> DescriptiveResult:
    """
    Analyse the relationship between exposure distribution and outcomes.

    Stratifies exposure into quartiles and computes outcome rates per
    stratum with a trend test (Cochran-Armitage-like via correlation).

    Parameters
    ----------
    exposure_levels : array-like
        Continuous exposure measurements.
    health_outcomes : array-like
        Binary health outcomes (0/1).
    method : str
        Stratification method: 'quartile' or 'median'.

    Returns
    -------
    DescriptiveResult
        extra has 'strata_rates', 'trend_r', 'trend_p'.

    References
    ----------
    Nieuwenhuijsen, M. J. (2015). *Exposure Assessment in
    Environmental Epidemiology*, 2nd ed. Oxford University Press.
    """
    exp = np.asarray(exposure_levels, dtype=float)
    out = np.asarray(health_outcomes, dtype=int)
    if len(exp) != len(out):
        raise ValueError("exposure and outcome arrays must match.")
    if len(exp) < 8:
        raise ValueError("Need at least 8 observations.")

    if method == "quartile":
        cuts = np.percentile(exp, [25, 50, 75])
        labels = np.digitize(exp, cuts)
    elif method == "median":
        labels = (exp >= np.median(exp)).astype(int)
    else:
        raise ValueError(f"Unknown method: {method}")

    strata_rates = {}
    for s in np.unique(labels):
        mask = labels == s
        strata_rates[int(s)] = float(np.mean(out[mask]))

    r, p = stats.spearmanr(exp, out)

    return DescriptiveResult(
        name="exposure_assessment",
        value=float(r),
        extra={
            "strata_rates": strata_rates,
            "trend_r": float(r),
            "trend_p": float(p),
            "method": method,
            "n": len(exp),
        },
    )


expos = exposure_assessment


def cheatsheet() -> str:
    return "exposure_assessment({}) -> Exposure distribution analysis."
