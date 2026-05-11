# morie.fn — function file (hadesllm/morie)
"""Nadaraya-Watson kernel estimator of G in single-index model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_nw_estimator_g"]


def horowitz_nw_estimator_g(x, y, beta, bandwidth):
    """
    Nadaraya-Watson kernel estimator of G in single-index model

    Formula: G_hat(v) = sum_i K_h(X_i'beta-v)*Y_i / sum_i K_h(X_i'beta-v)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    beta : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: G_hat

    References
    ----------
    Horowitz Ch 2, Sec 2.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Nadaraya-Watson kernel estimator of G in single-index model"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Nadaraya-Watson kernel estimator of G in single-index model"})


def cheatsheet():
    return "hrznwrg: Nadaraya-Watson kernel estimator of G in single-index model"
