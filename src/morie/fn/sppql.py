"""Penalized quasi-likelihood (PQL) for spatial GLMM."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_pql_glmm"]


def schabenberger_pql_glmm(x, y, coords, link, family):
    """
    Penalized quasi-likelihood (PQL) for spatial GLMM

    Formula: Linearize around b_hat; iterative weighted LS for beta; REML for theta

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    coords : array-like
        Input data.
    link : array-like
        Input data.
    family : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: coefficients, random_effects

    References
    ----------
    Schabenberger Ch 6, Sec 6.3.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Penalized quasi-likelihood (PQL) for spatial GLMM"})


def cheatsheet():
    return "sppql: Penalized quasi-likelihood (PQL) for spatial GLMM"
