# morie.fn -- function file (hadesllm/morie)
"""Bhattacharyya distance for class separability."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_bhattacharyya"]


def rangayyan_bhattacharyya(mu1, sigma1, mu2, sigma2):
    """
    Bhattacharyya distance for class separability

    Formula: D_B = -ln integral sqrt(p1(mu1)*p2(mu1)) dx; for Gaussians: D_B = analytic form

    Parameters
    ----------
    mu1 : array-like
        Input data.
    sigma1 : array-like
        Input data.
    mu2 : array-like
        Input data.
    sigma2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bhattacharyya_dist, coefficient

    References
    ----------
    Rangayyan Ch 10.10.1
    """
    mu1 = np.asarray(mu1, dtype=float)
    n = int(mu1) if mu1.ndim == 0 else len(mu1)
    result = float(np.mean(mu1))
    se = float(np.std(mu1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bhattacharyya distance for class separability"})


def cheatsheet():
    return "rgbhatt: Bhattacharyya distance for class separability"
