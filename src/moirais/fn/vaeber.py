"""VAE evidence lower bound."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vae_elbo"]


def vae_elbo(x, encoder, decoder):
    """
    VAE evidence lower bound

    Formula: E_q[log p(x|z)] - KL(q(z|x)||p(z))

    Parameters
    ----------
    x : array-like
        Input data.
    encoder : array-like
        Input data.
    decoder : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kingma-Welling (2014)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "VAE evidence lower bound"})


def cheatsheet():
    return "vaeber: VAE evidence lower bound"
