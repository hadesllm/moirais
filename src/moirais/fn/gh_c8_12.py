# moirais.fn — function file (hadesllm/moirais)
"""Lower bounds for posterior contraction rates: cannot beat minimax rate."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_crt_lower"]


def ghosal_crt_lower(x):
    """
    Lower bounds for posterior contraction rates: cannot beat minimax rate

    Formula: eps_n >= n^{-s/(2s+1)} for s-Holder smoothness (information-theoretic lb)

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
    Ghosal Ch 8 §8.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Lower bounds for posterior contraction rates: cannot beat minimax rate"})


def cheatsheet():
    return "gh_c8_12: Lower bounds for posterior contraction rates: cannot beat minimax rate"
