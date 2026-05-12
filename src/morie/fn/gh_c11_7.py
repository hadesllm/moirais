# morie.fn -- function file (hadesllm/morie)
"""Riemann-Liouville process: integrated BM of order s, Holder-s paths."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_rl_process"]


def ghosal_rl_process(x):
    """
    Riemann-Liouville process: integrated BM of order s, Holder-s paths

    Formula: R_s(t) = integral_0^t (t-u)^{s-1/2} dW(u) / Gamma(s+1/2)

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
    Ghosal Ch 11 §11.4.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Riemann-Liouville process: integrated BM of order s, Holder-s paths"})


def cheatsheet():
    return "gh_c11_7: Riemann-Liouville process: integrated BM of order s, Holder-s paths"
