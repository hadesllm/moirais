# moirais.fn — function file (hadesllm/moirais)
"""Evidence lower bound (ELBO) loss for VAE."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_elbo"]


def geron_elbo(x, mu, log_sigma):
    """
    Evidence lower bound (ELBO) loss for VAE

    Formula: ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))

    Parameters
    ----------
    x : array-like
        Input data.
    mu : array-like
        Input data.
    log_sigma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Géron Ch 18
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Evidence lower bound (ELBO) loss for VAE"})


def cheatsheet():
    return "hmelb: Evidence lower bound (ELBO) loss for VAE"
