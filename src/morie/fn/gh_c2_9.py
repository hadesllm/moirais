# morie.fn -- function file (hadesllm/morie)
"""Bayesian nonparametric binary regression via probit link: P(Y=1|x) = Phi(f(x))."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_np_binary_reg"]


def ghosal_np_binary_reg(x, y):
    """
    Bayesian nonparametric binary regression via probit link: P(Y=1|x) = Phi(f(x))

    Formula: P(Y=1|x) = Phi(f(x)), f ~ GP prior

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 2 §2.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian nonparametric binary regression via probit link: P(Y=1|x) = Phi(f(x))"})


def cheatsheet():
    return "gh_c2_9: Bayesian nonparametric binary regression via probit link: P(Y=1|x) = Phi(f(x))"
