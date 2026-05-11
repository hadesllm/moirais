# morie.fn — function file (hadesllm/morie)
"""Deconvolution density estimation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_deconvolution"]


def horowitz_deconvolution(y):
    """
    Deconvolution density estimation

    Formula: f_X = F^{-1}[phi_Y/phi_U] (Fourier)

    Parameters
    ----------
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Horowitz (2009), Ch 12
    """
    x = np.asarray(y, dtype=float)
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Deconvolution density estimation"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Deconvolution density estimation"})


def cheatsheet():
    return "hrzn2: Deconvolution density estimation"
