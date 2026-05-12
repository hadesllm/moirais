# morie.fn -- function file (hadesllm/morie)
"""GAN minimax objective for generator G and discriminator D."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_gan_minimax"]


def geron_gan_minimax(real, fake, D_real, D_fake):
    """
    GAN minimax objective for generator G and discriminator D

    Formula: min_G max_D E_x[log D(x)] + E_z[log(1 - D(G(z)))]

    Parameters
    ----------
    real : array-like
        Input data.
    fake : array-like
        Input data.
    D_real : array-like
        Input data.
    D_fake : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss_G, loss_D

    References
    ----------
    Géron Ch 18, GAN section (Goodfellow 2014)
    """
    real = np.atleast_1d(np.asarray(real, dtype=float))
    n = len(real)
    result = float(np.mean(real))
    se = float(np.std(real, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GAN minimax objective for generator G and discriminator D"})


def cheatsheet():
    return "grgan: GAN minimax objective for generator G and discriminator D"
