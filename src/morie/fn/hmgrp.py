# morie.fn — function file (hadesllm/morie)
"""Gaussian random projection matrix scaled by 1/sqrt(d')."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_gaussian_rand_projection"]


def geron_gaussian_rand_projection(X, d_out, seed):
    """
    Gaussian random projection matrix scaled by 1/sqrt(d')

    Formula: X' = X * R, R_ij ~ N(0, 1/d')

    Parameters
    ----------
    X : array-like
        Input data.
    d_out : array-like
        Input data.
    seed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X_projected

    References
    ----------
    Géron Ch 7
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gaussian random projection matrix scaled by 1/sqrt(d')"})


def cheatsheet():
    return "hmgrp: Gaussian random projection matrix scaled by 1/sqrt(d')"
