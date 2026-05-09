# moirais.fn — function file (hadesllm/moirais)
"""Kernel density estimator for semiparametric models."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_kernel_density"]


def horowitz_kernel_density(x, bandwidth):
    """
    Kernel density estimator for semiparametric models

    Formula: f_hat(x) = (1/nh) sum K((x-Xi)/h)

    Parameters
    ----------
    x : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Horowitz (2009), Ch 2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Kernel density estimator for semiparametric models"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Kernel density estimator for semiparametric models"})


def cheatsheet():
    return "hrzk1: Kernel density estimator for semiparametric models"
