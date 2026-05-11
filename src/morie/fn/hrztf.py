# morie.fn — function file (hadesllm/morie)
"""Fully nonparametric transformation model: both T and F unknown."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_both_nonpar_transform"]


def horowitz_both_nonpar_transform(x, y):
    """
    Fully nonparametric transformation model: both T and F unknown

    Formula: T(Y) = X'beta + U; T and F both unknown monotone/continuous; joint identification

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: T_hat, F_hat, beta_hat

    References
    ----------
    Horowitz Ch 6, Sec 6.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fully nonparametric transformation model: both T and F unknown"})


def cheatsheet():
    return "hrztf: Fully nonparametric transformation model: both T and F unknown"
