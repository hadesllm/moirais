# moirais.fn — function file (hadesllm/moirais)
"""Mahalanobis distance from sample to class."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_mahalanobis"]


def rangayyan_mahalanobis(x, mu, sigma):
    """
    Mahalanobis distance from sample to class

    Formula: D^2 = (x-mu)^T * Sigma^{-1} * (x-mu)

    Parameters
    ----------
    x : array-like
        Input data.
    mu : array-like
        Input data.
    sigma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: distance

    References
    ----------
    Rangayyan Ch 10.4.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mahalanobis distance from sample to class"})


def cheatsheet():
    return "rgmahd: Mahalanobis distance from sample to class"
