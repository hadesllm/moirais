# morie.fn -- function file (hadesllm/morie)
"""Indirect standardization (SMR-based adjusted rate)."""

from __future__ import annotations

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def indirect_standardization(
    observed: int,
    populations: list[int] | np.ndarray,
    reference_rates: list[float] | np.ndarray,
    crude_ref_rate: float | None = None,
    confidence: float = 0.95,
) -> ESRes:
    """Compute indirectly standardized rate via the SMR method.

    Expected count = sum(pop_i * ref_rate_i). SMR = observed / expected.
    Adjusted rate = SMR * crude reference rate.

    Parameters
    ----------
    observed : int
        Total observed events in the study population.
    populations : array-like of int
        Study population per age stratum.
    reference_rates : array-like of float
        Reference (standard) rates per stratum.
    crude_ref_rate : float, optional
        Crude rate in reference population. If None, computed as
        weighted mean of reference_rates.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    ESRes

    References
    ----------
    Rothman, K. J. et al. (2008). *Modern Epidemiology*, 3rd ed.
    Lippincott Williams & Wilkins, Ch. 4.
    """
    pops = np.asarray(populations, dtype=float)
    ref = np.asarray(reference_rates, dtype=float)

    if len(pops) != len(ref):
        raise ValueError("populations and reference_rates must match")
    if observed < 0:
        raise ValueError("observed must be non-negative")

    expected = float(np.sum(pops * ref))
    if expected <= 0:
        raise ValueError("Expected count must be positive")

    smr = observed / expected

    if crude_ref_rate is None:
        crude_ref_rate = float(np.mean(ref))

    adj_rate = smr * crude_ref_rate

    alpha = 1 - confidence
    if observed == 0:
        smr_lo = 0.0
        smr_hi = stats.chi2.ppf(1 - alpha / 2, 2) / (2 * expected)
    else:
        smr_lo = stats.chi2.ppf(alpha / 2, 2 * observed) / (2 * expected)
        smr_hi = stats.chi2.ppf(1 - alpha / 2, 2 * (observed + 1)) / (2 * expected)

    return ESRes(
        measure="indirect_std_rate",
        estimate=float(adj_rate),
        ci_lower=float(smr_lo * crude_ref_rate),
        ci_upper=float(smr_hi * crude_ref_rate),
        n=observed,
        extra={"SMR": float(smr), "expected": expected, "crude_ref_rate": crude_ref_rate},
    )


ismrt = indirect_standardization


def cheatsheet() -> str:
    return "indirect_standardization({}) -> Indirectly standardized rate (SMR method)."
