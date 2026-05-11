# morie.fn — function file (hadesllm/morie)
"""Mean of DP: E[G(A)] = G0(A) for all measurable A."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dp_mean"]


def ghosal_dp_mean(x):
    """
    Mean of DP: E[G(A)] = G0(A) for all measurable A

    Formula: E[G(A)] = G0(A)

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
    Ghosal Ch 4 §4.1.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mean of DP: E[G(A)] = G0(A) for all measurable A"})


def cheatsheet():
    return "gh_c4_2: Mean of DP: E[G(A)] = G0(A) for all measurable A"
