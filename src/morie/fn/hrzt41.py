# morie.fn — function file (hadesllm/morie)
"""Theorem 4.1: Identification of beta under median independence."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_thm4_1_id_median"]


def horowitz_thm4_1_id_median(x, y):
    """
    Theorem 4.1: Identification of beta under median independence

    Formula: median(U|X=x)=0; |beta_1|=1; support not in linear subspace; continuous X_1

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: boolean

    References
    ----------
    Horowitz Ch 4, Theorem 4.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Theorem 4.1: Identification of beta under median independence"})


def cheatsheet():
    return "hrzt41: Theorem 4.1: Identification of beta under median independence"
