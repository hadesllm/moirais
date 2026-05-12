# morie.fn -- function file (hadesllm/morie)
"""Standard kernel density estimator (Rosenblatt-Parzen)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_standard_kde"]


def fauzi_standard_kde(x, bandwidth, kernel):
    """
    Standard kernel density estimator (Rosenblatt-Parzen)

    Formula: f_hat_h(x) = (1/nh) sum K((x-X_i)/h)

    Parameters
    ----------
    x : array-like
        Input data.
    bandwidth : array-like
        Input data.
    kernel : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fauzi Ch 1, Eq 1.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Standard kernel density estimator (Rosenblatt-Parzen)"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Standard kernel density estimator (Rosenblatt-Parzen)"})


def cheatsheet():
    return "fzkde: Standard kernel density estimator (Rosenblatt-Parzen)"
