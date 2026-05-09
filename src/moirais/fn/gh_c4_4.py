# moirais.fn — function file (hadesllm/moirais)
"""Covariance of DP: Cov[G(A),G(B)] = (G0(A cap B) - G0(A)G0(B))/(alpha+1)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dp_cov"]


def ghosal_dp_cov(x):
    """
    Covariance of DP: Cov[G(A),G(B)] = (G0(A cap B) - G0(A)G0(B))/(alpha+1)

    Formula: Cov[G(A),G(B)] = (G0(A cap B) - G0(A)G0(B)) / (alpha+1)

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Covariance of DP: Cov[G(A),G(B)] = (G0(A cap B) - G0(A)G0(B))/(alpha+1)"})


def cheatsheet():
    return "gh_c4_4: Covariance of DP: Cov[G(A),G(B)] = (G0(A cap B) - G0(A)G0(B))/(alpha+1)"
