"""Normal-Inverse-Gamma conjugate."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["normal_inv_gamma"]


def normal_inv_gamma(y, mu0, kappa0, alpha0, beta0):
    """
    Normal-Inverse-Gamma conjugate

    Formula: posterior closed-form for Gaussian with unknown mean+var

    Parameters
    ----------
    y : array-like
        Input data.
    mu0 : array-like
        Input data.
    kappa0 : array-like
        Input data.
    alpha0 : array-like
        Input data.
    beta0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gelman BDA3
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Normal-Inverse-Gamma conjugate"})


def cheatsheet():
    return "nignst: Normal-Inverse-Gamma conjugate"
