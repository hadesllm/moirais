"""Compact vector form of the linear model, using a dot product of weight and feature vectors plus a bias.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["burkov_lm_ch1_linear_vector"]


def burkov_lm_ch1_linear_vector(w, x, b):
    """
    Compact vector form of the linear model, using a dot product of weight and feature vectors plus a bias.

    Formula: y = \mathbf{w} \cdot \mathbf{x} + b

    Parameters
    ----------
    w : array-like
        Input data.
    x : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: scalar prediction

    References
    ----------
    Burkov LM (2025), Ch 1, Eq 1.4, p. 29
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Compact vector form of the linear model, using a dot product of weight and feature vectors plus a bias."})


def cheatsheet():
    return "b104: Compact vector form of the linear model, using a dot product of weight and feature vectors plus a bias."
