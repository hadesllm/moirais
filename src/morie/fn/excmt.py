# morie.fn — function file (hadesllm/morie)
"""Excess mortality estimation via baseline comparison."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as _st


def excess_mortality(
    observed: np.ndarray,
    baseline: np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""Estimate excess mortality by comparing observed to baseline deaths.

    Excess deaths are the difference between observed and expected
    (baseline) deaths. The baseline may come from the average of
    preceding years or a model prediction.

    .. math::

        \\text{Excess} = \\sum_t (O_t - E_t)

        \\text{P-score}_t = \\frac{O_t - E_t}{E_t} \\times 100

    Parameters
    ----------
    observed : array_like
        Observed death counts per time period.
    baseline : array_like
        Expected (baseline) death counts per time period.
    alpha : float, default 0.05
        Significance level for the prediction interval.

    Returns
    -------
    dict
        Keys: 'total_excess', 'p_score_pct', 'excess_per_period',
              'z_scores', 'significant_periods', 'ci_lower', 'ci_upper'.

    References
    ----------
    Kontis, V. et al. (2020). Magnitude, demographics and dynamics of
    the effect of the first wave of the COVID-19 pandemic on all-cause
    mortality in 21 industrialised countries. Nature Medicine, 26(12),
    1919-1928.
    """
    obs = np.asarray(observed, dtype=float)
    base = np.asarray(baseline, dtype=float)

    if obs.shape != base.shape:
        raise ValueError("observed and baseline must have same shape.")
    if obs.ndim != 1:
        raise ValueError("Arrays must be 1-D.")

    excess = obs - base
    total_excess = float(np.sum(excess))
    total_baseline = float(np.sum(base))
    p_score = (total_excess / total_baseline * 100) if total_baseline > 0 else np.nan

    se_base = np.sqrt(base)
    z = _st.norm.ppf(1 - alpha / 2)
    ci_lo = base - z * se_base
    ci_hi = base + z * se_base

    z_scores = excess / np.where(se_base > 0, se_base, 1.0)
    sig = np.abs(z_scores) > z

    return {
        "total_excess": total_excess,
        "p_score_pct": float(p_score),
        "excess_per_period": excess,
        "z_scores": z_scores,
        "significant_periods": np.where(sig)[0],
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
    }


excmt = excess_mortality


def cheatsheet() -> str:
    return "excess_mortality({}) -> Excess mortality via baseline comparison."
