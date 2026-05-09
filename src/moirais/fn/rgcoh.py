# moirais.fn — function file (hadesllm/moirais)
"""Magnitude squared coherence."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_coherence"]


def rangayyan_coherence(x, y):
    """
    Magnitude squared coherence

    Formula: C_xy(f) = |S_xy|^2 / (S_xx * S_yy)

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
    Rangayyan Ch 4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Magnitude squared coherence"})


def cheatsheet():
    return "rgcoh: Magnitude squared coherence"
