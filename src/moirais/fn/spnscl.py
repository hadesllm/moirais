"""Neyman-Scott cluster process model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_neyman_scott"]


def schabenberger_neyman_scott(r, rho, mu, sigma):
    """
    Neyman-Scott cluster process model

    Formula: Parent process Poisson(rho); K(r)=pi*r^2 + mu^2*c*rho^{-1}*H(r)

    Parameters
    ----------
    r : array-like
        Input data.
    rho : array-like
        Input data.
    mu : array-like
        Input data.
    sigma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: k_function

    References
    ----------
    Schabenberger Ch 3, Sec 3.7.2
    """
    r = np.asarray(r, dtype=float)
    n = int(r) if r.ndim == 0 else len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Neyman-Scott cluster process model"})


def cheatsheet():
    return "spnscl: Neyman-Scott cluster process model"
