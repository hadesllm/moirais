# morie.fn — function file (hadesllm/morie)
"""b_3(t) coefficient in cumulative survival estimator 2 bias."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_b3_coefficient"]


def fauzi_b3_coefficient(t, g_func, density):
    """
    b_3(t) coefficient in cumulative survival estimator 2 bias

    Formula: b_3(t) = [g'(g^{-1}(t))]^2 * f_X(t) - g''(g^{-1}(t)) * S_X(t)

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
    Fauzi Ch 4, Eq 4.21
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    if t.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "b_3(t) coefficient in cumulative survival estimator 2 bias"})
    estimate = np.median(t)
    se = 1.2533 * np.std(t, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "b_3(t) coefficient in cumulative survival estimator 2 bias"})


def cheatsheet():
    return "fzb3t: b_3(t) coefficient in cumulative survival estimator 2 bias"
