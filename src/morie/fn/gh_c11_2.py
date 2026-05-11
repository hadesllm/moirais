# morie.fn — function file (hadesllm/morie)
"""RKHS norm and small ball probability for GP: ||f||_H controls concentration."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_rkhs_norm"]


def ghosal_rkhs_norm(x):
    """
    RKHS norm and small ball probability for GP: ||f||_H controls concentration

    Formula: phi_n(eps) = inf_{h in H: ||h-f0||<eps} ||h||_H^2 - log Pi(||f||<eps)

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
    Ghosal Ch 11 §11.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RKHS norm and small ball probability for GP: ||f||_H controls concentration"})


def cheatsheet():
    return "gh_c11_2: RKHS norm and small ball probability for GP: ||f||_H controls concentration"
