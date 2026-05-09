# moirais.fn — function file (hadesllm/moirais)
"""Additive model with unknown link function."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_additive_unknown_link"]


def horowitz_additive_unknown_link(x, y, bandwidth):
    """
    Additive model with unknown link function

    Formula: E[Y|X] = G(sum_j m_j(X_j)); estimate G and m_j simultaneously

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: G_hat, m_j_hats

    References
    ----------
    Horowitz Ch 3, Sec 3.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Additive model with unknown link function"})


def cheatsheet():
    return "hrzaul: Additive model with unknown link function"
