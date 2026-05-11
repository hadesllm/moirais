# morie.fn — function file (hadesllm/morie)
"""Direct age-adjustment rate."""

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def direct_age_adjustment(
    observed_rates: np.ndarray,
    pop_weights: np.ndarray,
    standard_weights: np.ndarray,
    confidence: float = 0.95,
    per: float = 100_000.0,
) -> ESRes:
    """Direct age-adjusted rate.

    Weights observed stratum-specific rates by a standard population
    distribution.

    .. math::

        DAR = \\frac{\\sum_i w^s_i \\cdot r_i}{\\sum_i w^s_i} \\times \\text{per}

    Parameters
    ----------
    observed_rates : array-like
        Stratum-specific rates (already per-unit, not per-100k).
    pop_weights : array-like
        Study population size per stratum (for variance).
    standard_weights : array-like
        Standard population weights per stratum.
    confidence : float, default 0.95
        Confidence level.
    per : float, default 100000
        Multiplier.

    Returns
    -------
    ESRes

    References
    ----------
    Fleiss, J. L., Levin, B., & Paik, M. C. (2003). Statistical Methods
    for Rates and Proportions. 3rd ed. Wiley.
    """
    rates = np.asarray(observed_rates, dtype=float)
    pw = np.asarray(pop_weights, dtype=float)
    sw = np.asarray(standard_weights, dtype=float)

    if len(rates) != len(pw) or len(rates) != len(sw):
        raise ValueError("All arrays must have equal length")

    w = sw / np.sum(sw)
    dar_val = np.sum(w * rates) * per

    var_dar = np.sum((w**2) * rates * (1 - rates) / pw) * per**2
    se = np.sqrt(max(var_dar, 0.0))
    z = stats.norm.ppf((1 + confidence) / 2)

    return ESRes(
        measure="DAR",
        estimate=float(dar_val),
        ci_lower=float(dar_val - z * se),
        ci_upper=float(dar_val + z * se),
        se=float(se),
    )


dar = direct_age_adjustment


def cheatsheet() -> str:
    return "direct_age_adjustment({}) -> Direct age-adjustment rate."
