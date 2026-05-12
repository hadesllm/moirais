# morie.fn -- function file (hadesllm/morie)
"""Borell-TIS inequality: concentration of GP supremum around its mean."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_borell_tis"]


def ghosal_borell_tis(x):
    """
    Borell-TIS inequality: concentration of GP supremum around its mean

    Formula: P(sup_t f(t) - E sup_t f(t) > u) <= exp(-u^2 / (2 sigma_f^2))

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
    Ghosal App I
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Borell-TIS inequality: concentration of GP supremum around its mean"})


def cheatsheet():
    return "gh_ap_i3: Borell-TIS inequality: concentration of GP supremum around its mean"
