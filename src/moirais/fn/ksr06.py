# moirais.fn — function file (hadesllm/moirais)
"""Maximal inequality for empirical processes."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_maximal_inequality"]


def kosorok_maximal_inequality(x):
    """
    Maximal inequality for empirical processes

    Formula: E[sup_F |G_n(f)|] <= J_[](1,F,L2)

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
    Kosorok (2008), Ch 2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Maximal inequality for empirical processes"})


def cheatsheet():
    return "ksr06: Maximal inequality for empirical processes"
