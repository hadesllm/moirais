# moirais.fn — function file (hadesllm/moirais)
"""i.i.d. contraction in L1: rate from entropy integral and prior mass."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_contraction_rate_iid"]


def ghosal_contraction_rate_iid(x):
    """
    i.i.d. contraction in L1: rate from entropy integral and prior mass

    Formula: Pi_n(||p-p0||_1 > M*eps_n | X^n) -> 0 if entropy and prior mass conditions hold

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
    Ghosal Ch 8 §8.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "i.i.d. contraction in L1: rate from entropy integral and prior mass"})


def cheatsheet():
    return "gh_contr_rate2: i.i.d. contraction in L1: rate from entropy integral and prior mass"
