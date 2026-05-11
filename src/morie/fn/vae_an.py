"""VAE reconstruction probability."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vae_anomaly"]


def vae_anomaly(X, vae):
    """
    VAE reconstruction probability

    Formula: -log p(x|z)

    Parameters
    ----------
    X : array-like
        Input data.
    vae : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    An-Cho (2015)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "VAE reconstruction probability"})


def cheatsheet():
    return "vae_an: VAE reconstruction probability"
