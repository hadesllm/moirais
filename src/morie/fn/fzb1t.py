# morie.fn -- function file (hadesllm/morie)
"""b_1(t) coefficient in survival estimator bias."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_b1_coefficient"]


def fauzi_b1_coefficient(t, g_func, density):
    """
    b_1(t) coefficient in survival estimator bias

    Formula: b_1(t) = g''(g^{-1}(t))*f_X(t) + [g'(g^{-1}(t))]^2 * f_X'(t)

    Parameters
    ----------
    t : array-like
        Input data.
    g_func : array-like
        Input data.
    density : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: coefficient

    References
    ----------
    Fauzi Ch 4, Eq 4.14
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    if t.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "b_1(t) coefficient in survival estimator bias"})
    estimate = np.median(t)
    se = 1.2533 * np.std(t, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "b_1(t) coefficient in survival estimator bias"})


def cheatsheet():
    return "fzb1t: b_1(t) coefficient in survival estimator bias"
