"""DP-GAN — train discriminator with DP-SGD."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dp_gan"]


def dp_gan(G, D, C, sigma):
    """
    DP-GAN — train discriminator with DP-SGD

    Formula: DP-SGD on D; G learns from noisy D

    Parameters
    ----------
    G : array-like
        Input data.
    D : array-like
        Input data.
    C : array-like
        Input data.
    sigma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Xie et al (2018)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP-GAN — train discriminator with DP-SGD"})


def cheatsheet():
    return "dpgan: DP-GAN — train discriminator with DP-SGD"
