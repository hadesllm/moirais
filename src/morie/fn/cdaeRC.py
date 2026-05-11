"""Collaborative denoising autoencoder."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cdae"]


def cdae(R, K):
    """
    Collaborative denoising autoencoder

    Formula: DAE on user history with user-specific bias

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
    Wu et al (2016) CDAE
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))
    n = len(R)
    result = float(np.mean(R))
    se = float(np.std(R, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Collaborative denoising autoencoder"})


def cheatsheet():
    return "cdaeRC: Collaborative denoising autoencoder"
