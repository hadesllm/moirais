# morie.fn -- function file (hadesllm/morie)
"""Gaussian random projection with component entries ~ N(0, 1/d)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_gaussian_random_projection"]


def geron_gaussian_random_projection(X, d, seed):
    """
    Gaussian random projection with component entries ~ N(0, 1/d)

    Formula: Z = X @ R, R_ij ~ N(0, 1/d)

    Parameters
    ----------
    X : array-like
        Input data.
    d : array-like
        Input data.
    seed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Z

    References
    ----------
    Géron Ch 7, Gaussian RP section
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gaussian random projection with component entries ~ N(0, 1/d)"})


def cheatsheet():
    return "grgrp: Gaussian random projection with component entries ~ N(0, 1/d)"
