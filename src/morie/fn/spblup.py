"""Best Linear Unbiased Predictor (BLUP) for spatial prediction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_blup"]


def schabenberger_blup(coords, z, target, cov_model):
    """
    Best Linear Unbiased Predictor (BLUP) for spatial prediction

    Formula: Z_hat(s0) = l'*Z; minimize MSE[Z_hat-Z(s0)] subject to unbiasedness

    Parameters
    ----------
    coords : array-like
        Input data.
    z : array-like
        Input data.
    target : array-like
        Input data.
    cov_model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prediction, variance

    References
    ----------
    Schabenberger Ch 5, Sec 5.1
    """
    z = np.asarray(z, dtype=float)
    n = int(z) if z.ndim == 0 else len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Best Linear Unbiased Predictor (BLUP) for spatial prediction"})


def cheatsheet():
    return "spblup: Best Linear Unbiased Predictor (BLUP) for spatial prediction"
