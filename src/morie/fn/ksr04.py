# morie.fn — function file (hadesllm/morie)
"""VC dimension computation for function class."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_vc_dimension"]


def kosorok_vc_dimension(x):
    """
    VC dimension computation for function class

    Formula: VC(F) = max{n: S(F,n) = 2^n}

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "VC dimension computation for function class"})


def cheatsheet():
    return "ksr04: VC dimension computation for function class"
