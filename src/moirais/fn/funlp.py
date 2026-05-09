# moirais.fn — function file (hadesllm/moirais)
"""Funnel plot data for publication bias. 'Look well into thyself; there is a source which will always spring up. — Marcus Aurelius' -- Ahsoka Tano"""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import DescriptiveResult


def funnel_plot(
    effects: np.ndarray,
    se: np.ndarray,
    confidence: float = 0.95,
) -> DescriptiveResult:
    """
    Compute funnel plot data and Egger's test for publication bias.

    The funnel plot displays effect sizes vs. their standard errors.
    Egger's regression tests for asymmetry (small-study effects).

    :param effects: Array of effect size estimates.
    :param se: Corresponding standard errors (same length, all > 0).
    :param confidence: Confidence level for the funnel. Default 0.95.
    :return: DescriptiveResult with Egger's test p-value.
    :raises ValueError: If arrays differ in length or se contains non-positive.

    References
    ----------
    Egger, M., Davey Smith, G., Schneider, M., & Minder, C. (1997). Bias
    in meta-analysis detected by a simple, graphical test. BMJ, 315(7109),
    629--634. doi:10.1136/bmj.315.7109.629
    """
    effects = np.asarray(effects, dtype=float)
    se = np.asarray(se, dtype=float)
    if effects.shape != se.shape or effects.ndim != 1 or effects.size < 3:
        raise ValueError("effects and se must be 1-D arrays of equal length >= 3.")
    if np.any(se <= 0):
        raise ValueError("All standard errors must be > 0.")

    w = 1.0 / (se**2)
    pooled = np.sum(w * effects) / np.sum(w)

    z_crit = _st.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    se_range = np.linspace(0.001, np.max(se) * 1.1, 100)
    upper = pooled + z_crit * se_range
    lower = pooled - z_crit * se_range

    precision = 1.0 / se
    std_effect = effects / se
    n_k = len(effects)
    slope, intercept, r_val, p_val, se_slope = _st.linregress(precision, std_effect)

    return DescriptiveResult(
        name="Funnel Plot / Egger's Test",
        value=float(np.round(p_val, 6)),
        extra={
            "pooled_effect": float(np.round(pooled, 4)),
            "egger_intercept": float(np.round(intercept, 4)),
            "egger_slope": float(np.round(slope, 4)),
            "egger_p_value": float(np.round(p_val, 6)),
            "funnel_se_range": se_range,
            "funnel_upper": upper,
            "funnel_lower": lower,
            "n_studies": n_k,
            "confidence": confidence,
        },
    )


funlp = funnel_plot


def cheatsheet() -> str:
    return "funnel_plot({}) -> Funnel plot data for publication bias. 'Look well into thyself; there is a source which will always spring up. — Marcus Aurelius' -- Ah"
