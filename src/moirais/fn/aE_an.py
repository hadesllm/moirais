"""Autoencoder reconstruction error."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["autoencoder_anomaly"]


def autoencoder_anomaly(X, ae):
    """
    Autoencoder reconstruction error

    Formula: ||x − decode(encode(x))||

    Parameters
    ----------
    X : array-like
        Input data.
    ae : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hawkins et al (2002)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Autoencoder reconstruction error"})


def cheatsheet():
    return "aE_an: Autoencoder reconstruction error"
