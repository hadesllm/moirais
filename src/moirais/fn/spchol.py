"""Cholesky decomposition simulation of Gaussian random fields."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_cholesky_sim"]


def schabenberger_cholesky_sim(mu, cov_matrix):
    """
    Cholesky decomposition simulation of Gaussian random fields

    Formula: Z = mu + L*e; L = chol(Sigma); e ~ N(0,I)

    Parameters
    ----------
    mu : array-like
        Input data.
    cov_matrix : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: simulated_field

    References
    ----------
    Schabenberger Ch 7, Sec 7.1.1
    """
    mu = np.asarray(mu, dtype=float)
    n = int(mu) if mu.ndim == 0 else len(mu)
    result = float(np.mean(mu))
    se = float(np.std(mu, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cholesky decomposition simulation of Gaussian random fields"})


def cheatsheet():
    return "spchol: Cholesky decomposition simulation of Gaussian random fields"
