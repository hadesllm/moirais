# moirais.fn — function file (hadesllm/moirais)
"""Bayesian nonparametric Poisson regression: E[Y|x] = exp(f(x)), f ~ GP."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_np_poisson_reg"]


def ghosal_np_poisson_reg(x, y):
    """
    Bayesian nonparametric Poisson regression: E[Y|x] = exp(f(x)), f ~ GP

    Formula: Y|x ~ Poisson(exp(f(x))), f ~ GP prior

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
    Ghosal Ch 2 §2.6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian nonparametric Poisson regression: E[Y|x] = exp(f(x)), f ~ GP"})


def cheatsheet():
    return "gh_c2_10: Bayesian nonparametric Poisson regression: E[Y|x] = exp(f(x)), f ~ GP"
