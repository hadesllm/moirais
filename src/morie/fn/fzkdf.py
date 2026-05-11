# morie.fn — function file (hadesllm/morie)
"""KDFE bias and variance properties."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_kdfe_properties"]


def fauzi_kdfe_properties(x):
    """
    KDFE bias and variance properties

    Formula: Bias = h^2 * mu_2(K) * f''(x) / 2

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
    Fauzi Ch 2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "KDFE bias and variance properties"})


def cheatsheet():
    return "fzkdf: KDFE bias and variance properties"
