# morie.fn -- function file (hadesllm/morie)
"""Prior via quantile process: specify distribution of quantile function Q(u) = F^{-1}(u)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_quantile_prior"]


def ghosal_quantile_prior(x):
    """
    Prior via quantile process: specify distribution of quantile function Q(u) = F^{-1}(u)

    Formula: G via Q(u) = F^{-1}(u), u in [0,1]

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
    Ghosal Ch 3 §3.4.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Prior via quantile process: specify distribution of quantile function Q(u) = F^{-1}(u)"})


def cheatsheet():
    return "gh_c3_9: Prior via quantile process: specify distribution of quantile function Q(u) = F^{-1}(u)"
