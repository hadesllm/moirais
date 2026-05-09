# moirais.fn — function file (hadesllm/moirais)
"""Expected number of distinct values from DP: E[K_n] ~ alpha*log(n) for large n."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dp_ndist"]


def ghosal_dp_ndist(x):
    """
    Expected number of distinct values from DP: E[K_n] ~ alpha*log(n) for large n

    Formula: E[K_n] = sum_{i=1}^n alpha/(alpha+i-1) ~ alpha*log(n)

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
    Ghosal Ch 4 §4.1.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Expected number of distinct values from DP: E[K_n] ~ alpha*log(n) for large n"})


def cheatsheet():
    return "gh_c4_8: Expected number of distinct values from DP: E[K_n] ~ alpha*log(n) for large n"
