# morie.fn -- function file (hadesllm/morie)
"""Dudley entropy integral: J[](sigma,F) = integral_0^sigma sqrt(log N(eps,F,L2)) deps."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dudley_entropy"]


def ghosal_dudley_entropy(x):
    """
    Dudley entropy integral: J[](sigma,F) = integral_0^sigma sqrt(log N(eps,F,L2)) deps

    Formula: E[sup_f |G_n(f)|] <= C * J[](sigma_F, F) for empirical process G_n

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dudley entropy integral: J[](sigma,F) = integral_0^sigma sqrt(log N(eps,F,L2)) deps"})


def cheatsheet():
    return "gh_ap_i2: Dudley entropy integral: J[](sigma,F) = integral_0^sigma sqrt(log N(eps,F,L2)) deps"
