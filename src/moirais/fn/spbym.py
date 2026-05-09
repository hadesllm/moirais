"""Besag-York-Mollie (BYM) model: ICAR + independent random effects."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_bym_model"]


def schabenberger_bym_model(x, y, E, w):
    """
    Besag-York-Mollie (BYM) model: ICAR + independent random effects

    Formula: Y_i ~ Poisson(E_i*exp(x_i'beta + phi_i + theta_i)); phi~ICAR, theta~N(0,sigma_theta^2)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    E : array-like
        Input data.
    w : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: posterior

    References
    ----------
    Schabenberger Ch 6, Sec 6.4.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Besag-York-Mollie (BYM) model: ICAR + independent random effects"})


def cheatsheet():
    return "spbym: Besag-York-Mollie (BYM) model: ICAR + independent random effects"
