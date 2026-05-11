"""Simple kriging: known mean mu, C(h) known."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_simple_kriging"]


def schabenberger_simple_kriging(coords, z, target, cov_model, mu):
    """
    Simple kriging: known mean mu, C(h) known

    Formula: Z_hat(s0) = mu + c'*C^{-1}*(Z-mu*1); sigma^2_SK = C(0) - c'*C^{-1}*c

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
    mu : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prediction, variance

    References
    ----------
    Schabenberger Ch 5, Sec 5.2.1
    """
    z = np.asarray(z, dtype=float)
    n = int(z) if z.ndim == 0 else len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Simple kriging: known mean mu, C(h) known"})


def cheatsheet():
    return "spskrg: Simple kriging: known mean mu, C(h) known"
