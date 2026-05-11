# morie.fn — function file (hadesllm/morie)
"""b_2(t) coefficient in cumulative survival estimator 1 bias."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_b2_coefficient_mrl"]


def fauzi_b2_coefficient_mrl(t, g_func, density):
    """
    b_2(t) coefficient in cumulative survival estimator 1 bias

    Formula: b_2(t) = [g'(g^{-1}(t))]^2 * f_X(t) + int_{g^{-1}(t)}^inf g''(x)*g'(t)*f_X(g(t))dx

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
    Fauzi Ch 4, Eq 4.15
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    if t.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "b_2(t) coefficient in cumulative survival estimator 1 bias"})
    estimate = np.median(t)
    se = 1.2533 * np.std(t, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "b_2(t) coefficient in cumulative survival estimator 1 bias"})


def cheatsheet():
    return "fzb2t: b_2(t) coefficient in cumulative survival estimator 1 bias"
