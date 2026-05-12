# morie.fn — function file (hadesllm/morie)
"""Direct standardization (age-adjusted rates)."""

from __future__ import annotations

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def direct_standardization(
    deaths: list[int] | np.ndarray,
    populations: list[int] | np.ndarray,
    standard_pop: list[float] | np.ndarray,
    confidence: float = 0.95,
) -> ESRes:
    r"""Compute directly age-standardized rate.

    .. math::

        ASR = \\sum_i w_i \\cdot r_i

    where :math:`r_i` is the stratum-specific rate and :math:`w_i`
    is the standard population weight.

    Parameters
    ----------
    deaths : array-like of int
        Events per age stratum.
    populations : array-like of int
        Population per age stratum.
    standard_pop : array-like of float
        Standard population weights (will be normalized).
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    ESRes

    References
    ----------
    Breslow, N. E. & Day, N. E. (1987). *Statistical Methods in
    Cancer Research*, Vol. II. IARC Scientific Publications No. 82.
    """
    d = np.asarray(deaths, dtype=float)
    n = np.asarray(populations, dtype=float)
    w = np.asarray(standard_pop, dtype=float)

    if len(d) != len(n) or len(d) != len(w):
        raise ValueError("All arrays must have the same length")
    if np.any(n <= 0):
        raise ValueError("populations must be positive")

    w = w / w.sum()
    rates = d / n
    asr = float(np.sum(w * rates))

    var_asr = float(np.sum(w**2 * d / n**2))
    se = np.sqrt(var_asr)
    z = stats.norm.ppf((1 + confidence) / 2)

    return ESRes(
        measure="direct_std_rate",
        estimate=asr,
        se=float(se),
        ci_lower=float(asr - z * se),
        ci_upper=float(asr + z * se),
        n=int(np.sum(d)),
        extra={"stratum_rates": rates.tolist(), "weights": w.tolist()},
    )


dsmrt = direct_standardization


def cheatsheet() -> str:
    return "direct_standardization({}) -> Directly age-standardized rate."
