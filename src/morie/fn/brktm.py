# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Berkson's bias diagnostic test."""

from __future__ import annotations

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def berkson_bias_test(or_hospital: float, or_population: float, se_hospital: float, se_population: float, confidence: float = 0.95, cdf=None) -> ESRes:
    """Test for Berkson's bias by comparing hospital vs population ORs.

    Berkson's bias arises in hospital-based case-control studies when
    both exposure and disease independently affect hospitalization.
    A significant difference between hospital and population ORs
    suggests Berkson's bias.

    Parameters
    ----------
    or_hospital : float
        Odds ratio from hospital-based study.
    or_population : float
        Odds ratio from population-based study.
    se_hospital : float
        Standard error of log(OR) from hospital study.
    se_population : float
        Standard error of log(OR) from population study.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    ESRes

    References
    ----------
    Berkson, J. (1946). Limitations of the application of fourfold
    table analysis to hospital data. Biometrics Bulletin, 2(3), 47-53.
    """
    if or_hospital <= 0 or or_population <= 0:
        raise ValueError("ORs must be positive")
    if se_hospital <= 0 or se_population <= 0:
        raise ValueError("SEs must be positive")

    ln_diff = np.log(or_hospital) - np.log(or_population)
    se_diff = np.sqrt(se_hospital**2 + se_population**2)
    z_stat = ln_diff / se_diff
    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    z = stats.norm.ppf((1 + confidence) / 2)
    ratio = or_hospital / or_population

    return ESRes(
        measure="berkson_bias",
        estimate=float(ratio),
        se=float(se_diff),
        ci_lower=float(np.exp(ln_diff - z * se_diff)),
        ci_upper=float(np.exp(ln_diff + z * se_diff)),
        extra={
            "z_statistic": float(z_stat),
            "p_value": float(p_val),
            "bias_detected": p_val < 0.05,
            "or_hospital": or_hospital,
            "or_population": or_population,
        },
    )


brktm = berkson_bias_test


def cheatsheet() -> str:
    return "berkson_bias_test({}) -> Berkson's bias diagnostic (hospital vs population OR)."
