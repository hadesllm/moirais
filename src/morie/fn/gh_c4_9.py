# morie.fn -- function file (hadesllm/morie)
"""Gamma process construction of DP: G(A) = gamma(A)/gamma(X)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dp_gamma"]


def ghosal_dp_gamma(x):
    """
    Gamma process construction of DP: G(A) = gamma(A)/gamma(X)

    Formula: G(A) = Gamma_A / Gamma_X, Gamma_A ~ Ga(alpha*G0(A),1)

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
    Ghosal Ch 4 §4.2.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gamma process construction of DP: G(A) = gamma(A)/gamma(X)"})


def cheatsheet():
    return "gh_c4_9: Gamma process construction of DP: G(A) = gamma(A)/gamma(X)"
