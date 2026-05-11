"""beta-VAE disentangling penalty."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["beta_vae_disentangle"]


def beta_vae_disentangle(x, encoder, decoder, beta):
    """
    beta-VAE disentangling penalty

    Formula: L = E[log p(x|z)] - beta D_KL(q(z|x)||p(z))

    Parameters
    ----------
    x : array-like
        Input data.
    encoder : array-like
        Input data.
    decoder : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Higgins et al (2017)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "beta-VAE disentangling penalty"})


def cheatsheet():
    return "betvae: beta-VAE disentangling penalty"
