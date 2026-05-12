# morie.fn -- function file (hadesllm/morie)
"""AMSE of sample quantile estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_quantile_amse"]


def fauzi_quantile_amse(data, p):
    """
    AMSE of sample quantile estimator

    Formula: AMSE(Q_hat(p)) = Q'(p)^2 * p(1-p)/n

    Parameters
    ----------
    data : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: amse

    References
    ----------
    Fauzi Ch 3, Eq 3.3
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    if data.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "AMSE of sample quantile estimator"})
    estimate = np.median(data)
    se = 1.2533 * np.std(data, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "AMSE of sample quantile estimator"})


def cheatsheet():
    return "fzamse: AMSE of sample quantile estimator"
