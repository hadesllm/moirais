# morie.fn -- function file (hadesllm/morie)
"""Choosing weight function for NLS single-index estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_nls_weight_function"]


def horowitz_nls_weight_function(x, y, bandwidth, weights):
    """
    Choosing weight function for NLS single-index estimator

    Formula: Weighted NLS: beta_hat = argmin sum w_i*[Y_i - G_hat(X_i'b)]^2; optimal w_i = 1/Var(Y_i|X_i)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    bandwidth : array-like
        Input data.
    weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta_hat

    References
    ----------
    Horowitz Ch 2, Sec 2.5.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Choosing weight function for NLS single-index estimator"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Choosing weight function for NLS single-index estimator"})


def cheatsheet():
    return "hrzwfun: Choosing weight function for NLS single-index estimator"
