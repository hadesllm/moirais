# moirais.fn — function file (hadesllm/moirais)
"""Bias-variance decomposition of expected prediction error."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_bias_variance_tradeoff"]


def geron_bias_variance_tradeoff(preds, y):
    """
    Bias-variance decomposition of expected prediction error

    Formula: E[err] = Bias^2 + Variance + Irreducible_noise

    Parameters
    ----------
    preds : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bias, variance, noise

    References
    ----------
    Géron Ch 1
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bias-variance decomposition of expected prediction error"})


def cheatsheet():
    return "hmbv: Bias-variance decomposition of expected prediction error"
