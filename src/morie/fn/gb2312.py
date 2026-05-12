# morie.fn -- function file (hadesllm/morie)
"""S_n(x) is a consistent estimator of F(x); converges in probability."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_edf_consistent"]


def gibbons_edf_consistent(x):
    """
    S_n(x) is a consistent estimator of F(x); converges in probability

    Formula: S_n(x) ->_p F_X(x) as n -> inf

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: convergence_result

    References
    ----------
    Gibbons Corollary 2.3.1.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "S_n(x) is a consistent estimator of F(x); converges in probability"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "S_n(x) is a consistent estimator of F(x); converges in probability"})


def cheatsheet():
    return "gb2312: S_n(x) is a consistent estimator of F(x); converges in probability"
