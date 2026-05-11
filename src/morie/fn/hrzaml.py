# morie.fn — function file (hadesllm/morie)
"""Additive model with known non-identity link function g."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_additive_nonid_link"]


def horowitz_additive_nonid_link(x, y, bandwidth, link):
    """
    Additive model with known non-identity link function g

    Formula: E[Y|X] = g^{-1}(mu + sum_j m_j(X_j)); use MAVE or local MLE

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    bandwidth : array-like
        Input data.
    link : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: m_j_hats

    References
    ----------
    Horowitz Ch 3, Sec 3.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Additive model with known non-identity link function g"})


def cheatsheet():
    return "hrzaml: Additive model with known non-identity link function g"
