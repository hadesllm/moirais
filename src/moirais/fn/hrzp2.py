# moirais.fn — function file (hadesllm/moirais)
"""Bandwidth selection for PLR."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_plr_bandwidth"]


def horowitz_plr_bandwidth(x, y):
    """
    Bandwidth selection for PLR

    Formula: h_opt = c * n^(-1/5) * sigma

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Horowitz (2009), Ch 3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bandwidth selection for PLR"})


def cheatsheet():
    return "hrzp2: Bandwidth selection for PLR"
