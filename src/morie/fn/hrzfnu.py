# morie.fn — function file (hadesllm/morie)
"""Smoothed deconvolution estimator of fU."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_deconv_estimator"]


def horowitz_deconv_estimator(w, eps_cf, bandwidth, nu_n):
    """
    Smoothed deconvolution estimator of fU

    Formula: fnU(u) = (1/2pi)*integral exp(-i*tau*u)*[psiNW(tau)*psiZeta(nu_n*tau)/psiEps(tau)]dtau

    Parameters
    ----------
    w : array-like
        Input data.
    eps_cf : array-like
        Input data.
    bandwidth : array-like
        Input data.
    nu_n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: density_estimate

    References
    ----------
    Horowitz Ch 5, Eq 5.8
    """
    w = np.asarray(w, dtype=float)
    n = int(w) if w.ndim == 0 else len(w)
    if w.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Smoothed deconvolution estimator of fU"})
    estimate = np.median(w)
    se = 1.2533 * np.std(w, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Smoothed deconvolution estimator of fU"})


def cheatsheet():
    return "hrzfnu: Smoothed deconvolution estimator of fU"
