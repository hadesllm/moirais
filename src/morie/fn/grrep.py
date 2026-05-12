# morie.fn -- function file (hadesllm/morie)
"""Reparameterization: z = mu + sigma * eps, eps ~ N(0, I)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_reparameterization_trick"]


def geron_reparameterization_trick(mu, logvar):
    """
    Reparameterization: z = mu + sigma * eps, eps ~ N(0, I)

    Formula: z = mu + exp(0.5 * logvar) * eps; eps ~ N(0, I)

    Parameters
    ----------
    mu : array-like
        Input data.
    logvar : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: z

    References
    ----------
    Géron Ch 18, Reparameterization Trick (VAE)
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    n = len(mu)
    result = float(np.mean(mu))
    se = float(np.std(mu, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Reparameterization: z = mu + sigma * eps, eps ~ N(0, I)"})


def cheatsheet():
    return "grrep: Reparameterization: z = mu + sigma * eps, eps ~ N(0, I)"
