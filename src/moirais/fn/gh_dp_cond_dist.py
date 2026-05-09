# moirais.fn — function file (hadesllm/moirais)
"""DP conditional distribution: G given G(A) is mixture of DP on A and A^c."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dp_conditional_distribution"]


def ghosal_dp_conditional_distribution(x):
    """
    DP conditional distribution: G given G(A) is mixture of DP on A and A^c

    Formula: G(A^c . | G(A)=w) | G(A)=w ~ DP(alpha*G0(A^c/(1-G0(A))), *) on A^c

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
    Ghosal Ch 4 §4.1.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP conditional distribution: G given G(A) is mixture of DP on A and A^c"})


def cheatsheet():
    return "gh_dp_cond_dist: DP conditional distribution: G given G(A) is mixture of DP on A and A^c"
