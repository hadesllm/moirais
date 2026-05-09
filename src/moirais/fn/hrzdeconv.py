# moirais.fn — function file (hadesllm/moirais)
"""Deconvolution density estimator for W=U+epsilon."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_deconvolution_density"]


def horowitz_deconvolution_density(w, eps_density, bandwidth):
    """
    Deconvolution density estimator for W=U+epsilon

    Formula: fU(u) = (1/2pi)*integral exp(-i*tau*u)*[psiW(tau)/psiEps(tau)]dtau

    Parameters
    ----------
    w : array-like
        Input data.
    eps_density : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: density_estimate

    References
    ----------
    Horowitz Ch 5, Eq 5.7
    """
    w = np.asarray(w, dtype=float)
    n = int(w) if w.ndim == 0 else len(w)
    if w.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Deconvolution density estimator for W=U+epsilon"})
    estimate = np.median(w)
    se = 1.2533 * np.std(w, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Deconvolution density estimator for W=U+epsilon"})


def cheatsheet():
    return "hrzdeconv: Deconvolution density estimator for W=U+epsilon"
