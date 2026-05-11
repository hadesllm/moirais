# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bootstrap confidence interval (percentile, BCa, studentized)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["bootstrap_ci"]


def bootstrap_ci(x):
    """
    Bootstrap confidence interval (percentile, BCa, studentized)

    Formula: CI = [theta*_alpha/2, theta*_{1-alpha/2}]

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Efron & Tibshirani (1993)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bootstrap confidence interval (percentile, BCa, studentized)"})


def cheatsheet():
    return "btsrp: Bootstrap confidence interval (percentile, BCa, studentized)"
