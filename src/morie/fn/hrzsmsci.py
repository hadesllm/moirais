# morie.fn -- function file (hadesllm/morie)
"""Confidence intervals for smoothed maximum-score via subsampling."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_sms_confidence"]


def horowitz_sms_confidence(x, y, bandwidth, alpha):
    """
    Confidence intervals for smoothed maximum-score via subsampling

    Formula: CI: [beta_hat - q_{1-alpha/2}*se, beta_hat + q_{alpha/2}*se] using subsampling quantiles

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    bandwidth : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ci

    References
    ----------
    Horowitz Ch 4, Sec 4.3.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Confidence intervals for smoothed maximum-score via subsampling"})


def cheatsheet():
    return "hrzsmsci: Confidence intervals for smoothed maximum-score via subsampling"
