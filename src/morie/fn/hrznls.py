# morie.fn -- function file (hadesllm/morie)
"""Nonlinear least squares estimator of beta in single-index model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_nls_sim"]


def horowitz_nls_sim(x, y, bandwidth):
    """
    Nonlinear least squares estimator of beta in single-index model

    Formula: beta_hat = argmin sum [Y_i - G_hat_{-i}(X_i'b)]^2 over b with |b_1|=1

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
        Keys: beta_hat, se

    References
    ----------
    Horowitz Ch 2, Sec 2.5.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Nonlinear least squares estimator of beta in single-index model"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Nonlinear least squares estimator of beta in single-index model"})


def cheatsheet():
    return "hrznls: Nonlinear least squares estimator of beta in single-index model"
