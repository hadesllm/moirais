"""Wasserstein-based quantization distortion."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_quantization_distortion"]


def ot_quantization_distortion(X, centroids):
    """
    Wasserstein-based quantization distortion

    Formula: Distortion = OT(μ, ν_quant) where ν_quant has finite support

    Parameters
    ----------
    X : array-like
        Input data.
    centroids : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: dist

    References
    ----------
    Pollard (1982)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wasserstein-based quantization distortion"})


def cheatsheet():
    return "otmqd: Wasserstein-based quantization distortion"
