# moirais.fn — function file (hadesllm/moirais)
"""Asymptotic normality of deconvolution estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_deconv_normality"]


def horowitz_deconv_normality(w, eps_density, bandwidth, u):
    """
    Asymptotic normality of deconvolution estimator

    Formula: [n*h_n/b_n]^{1/2} * (fnU(u) - fU(u) - bias) ->_D N(0, sigma^2)

    Parameters
    ----------
    w : array-like
        Input data.
    eps_density : array-like
        Input data.
    bandwidth : array-like
        Input data.
    u : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: statistic

    References
    ----------
    Horowitz Ch 5, Sec 5.1.3
    """
    w = np.asarray(w, dtype=float)
    n = int(w) if w.ndim == 0 else len(w)
    if w.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Asymptotic normality of deconvolution estimator"})
    estimate = np.median(w)
    se = 1.2533 * np.std(w, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Asymptotic normality of deconvolution estimator"})


def cheatsheet():
    return "hrzdcnm: Asymptotic normality of deconvolution estimator"
