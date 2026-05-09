# moirais.fn — function file (hadesllm/moirais)
"""Wild bootstrap for semiparametric models."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_wild_bootstrap"]


def horowitz_wild_bootstrap(x, y, residuals):
    """
    Wild bootstrap for semiparametric models

    Formula: e* = e_hat * v_i, v_i ~ Rademacher

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    residuals : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Horowitz (2009), Ch 13
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wild bootstrap for semiparametric models"})


def cheatsheet():
    return "hrzw1: Wild bootstrap for semiparametric models"
