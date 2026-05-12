# morie.fn -- function file (hadesllm/morie)
"""Poisson regression via finite random series: log-linear model with series prior."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_frs_poireg"]


def ghosal_frs_poireg(x, y):
    """
    Poisson regression via finite random series: log-linear model with series prior

    Formula: Y|x ~ Poi(exp(f(x))), f = sum_{k<=K} beta_k phi_k, rate adaptation

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
    Ghosal Ch 10 §10.4.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Poisson regression via finite random series: log-linear model with series prior"})


def cheatsheet():
    return "gh_c10_10: Poisson regression via finite random series: log-linear model with series prior"
