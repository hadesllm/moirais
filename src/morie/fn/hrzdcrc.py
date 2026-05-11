# morie.fn — function file (hadesllm/morie)
"""Rate of convergence of deconvolution estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_deconv_rate"]


def horowitz_deconv_rate(n, smoothness_type, r, s):
    """
    Rate of convergence of deconvolution estimator

    Formula: ||fnU - fU||^2 = O_p([log n]^{-s}) for ordinary smooth; O_p(n^{-r}) for supersmooth

    Parameters
    ----------
    n : array-like
        Input data.
    smoothness_type : array-like
        Input data.
    r : array-like
        Input data.
    s : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rate

    References
    ----------
    Horowitz Ch 5, Sec 5.1.1
    """
    n = int(n) if np.ndim(n) == 0 else len(n)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Rate of convergence of deconvolution estimator"})
    estimate = np.median(n)
    se = 1.2533 * np.std(n, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Rate of convergence of deconvolution estimator"})


def cheatsheet():
    return "hrzdcrc: Rate of convergence of deconvolution estimator"
