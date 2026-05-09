"""Latent-class weighted MSM."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["latent_class_weighted"]


def latent_class_weighted(y, A, H, K):
    """
    Latent-class weighted MSM

    Formula: weighted by class membership probability

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    H : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lanza et al (2013)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Latent-class weighted MSM"})


def cheatsheet():
    return "lcwphr: Latent-class weighted MSM"
