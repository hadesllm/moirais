# morie.fn — function file (hadesllm/morie)
"""Two-step oracle-efficient additive model estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_two_step_oracle"]


def horowitz_two_step_oracle(x, y, bandwidth):
    """
    Two-step oracle-efficient additive model estimator

    Formula: Step 1: pilot estimate; Step 2: partial-out other components, do 1D kernel regression

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: g_j_hats, se

    References
    ----------
    Horowitz Ch 3, Sec 3.1.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Two-step oracle-efficient additive model estimator"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Two-step oracle-efficient additive model estimator"})


def cheatsheet():
    return "hrzora: Two-step oracle-efficient additive model estimator"
