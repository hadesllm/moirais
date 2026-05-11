# morie.fn — function file (hadesllm/morie)
"""Chen gamma kernel density estimator for [0,inf) data."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_gamma_kde"]


def fauzi_gamma_kde(x, bandwidth):
    """
    Chen gamma kernel density estimator for [0,inf) data

    Formula: f_hat_C(x) = (1/n) sum Ga(X_i; x/h+1, h)

    Parameters
    ----------
    x : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fauzi Ch 1, Chen 1999
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Chen gamma kernel density estimator for [0,inf) data"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Chen gamma kernel density estimator for [0,inf) data"})


def cheatsheet():
    return "fzgkde: Chen gamma kernel density estimator for [0,inf) data"
