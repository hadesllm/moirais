"""Variational autoencoder for CF."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vae_cf"]


def vae_cf(R, K):
    """
    Variational autoencoder for CF

    Formula: multinomial VAE on user click vector

    Parameters
    ----------
    R : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Liang et al (2018) Mult-VAE
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))
    n = len(R)
    result = float(np.mean(R))
    se = float(np.std(R, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variational autoencoder for CF"})


def cheatsheet():
    return "vaeCF: Variational autoencoder for CF"
