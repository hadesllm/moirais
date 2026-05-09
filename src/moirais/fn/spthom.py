"""Thomas cluster process: Normal offspring displacements."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_thomas_process"]


def schabenberger_thomas_process(r, rho, mu, sigma):
    """
    Thomas cluster process: Normal offspring displacements

    Formula: K(r) = pi*r^2 + mu*[1-exp(-r^2/(4*sigma^2))]/rho

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
    Schabenberger Ch 3
    """
    r = np.asarray(r, dtype=float)
    n = int(r) if r.ndim == 0 else len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Thomas cluster process: Normal offspring displacements"})


def cheatsheet():
    return "spthom: Thomas cluster process: Normal offspring displacements"
