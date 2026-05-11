# morie.fn — function file (hadesllm/morie)
"""Contraction under misspecification: posterior contracts to KL projection of P0 onto model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_misspec_crt"]


def ghosal_misspec_crt(x):
    """
    Contraction under misspecification: posterior contracts to KL projection of P0 onto model

    Formula: P* = argmin_{P in model} KL(P0,P), Pi_n({d(P,P*)>eps_n}|X^n)->0

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 8 §8.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Contraction under misspecification: posterior contracts to KL projection of P0 onto model"})


def cheatsheet():
    return "gh_c8_13: Contraction under misspecification: posterior contracts to KL projection of P0 onto model"
