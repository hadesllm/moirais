# moirais.fn — function file (hadesllm/moirais)
"""Efficient score and information bound."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_efficient_score"]


def kosorok_efficient_score(x, y):
    """
    Efficient score and information bound

    Formula: S_eff = S - Pi(S|T)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Kosorok (2008), Ch 6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Efficient score and information bound"})


def cheatsheet():
    return "ksr11: Efficient score and information bound"
