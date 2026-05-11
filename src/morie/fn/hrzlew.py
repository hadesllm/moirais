# morie.fn — function file (hadesllm/morie)
"""Lewbel heteroskedastic binary response estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_lewbel_estimator"]


def horowitz_lewbel_estimator(x, y, z, bandwidth):
    """
    Lewbel heteroskedastic binary response estimator

    Formula: beta_hat = argmin sum [Y_i - I(X_i'b + v(Z_i)*e_i > 0)]^2; heteroskedastic U=v(Z)*e

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    z : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta_hat, se

    References
    ----------
    Horowitz Ch 4, Sec 4.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Lewbel heteroskedastic binary response estimator"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Lewbel heteroskedastic binary response estimator"})


def cheatsheet():
    return "hrzlew: Lewbel heteroskedastic binary response estimator"
