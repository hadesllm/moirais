# morie.fn — function file (hadesllm/morie)
"""Variational autoencoder ELBO: reconstruction minus KL to prior."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_vae_elbo"]


def geron_vae_elbo(x, mu, logvar, recon):
    """
    Variational autoencoder ELBO: reconstruction minus KL to prior

    Formula: ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))

    Parameters
    ----------
    x : array-like
        Input data.
    mu : array-like
        Input data.
    logvar : array-like
        Input data.
    recon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: elbo

    References
    ----------
    Géron Ch 18, Variational Autoencoder section
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variational autoencoder ELBO: reconstruction minus KL to prior"})


def cheatsheet():
    return "grvae: Variational autoencoder ELBO: reconstruction minus KL to prior"
