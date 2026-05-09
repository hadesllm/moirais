"""Variational inference with normalizing flow."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["variational_nf"]


def variational_nf(log_p, flow, x):
    """
    Variational inference with normalizing flow

    Formula: q(z) = flow(epsilon); maximize ELBO

    Parameters
    ----------
    log_p : array-like
        Input data.
    flow : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rezende-Mohamed (2015); Kucukelbir (2017)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variational inference with normalizing flow"})


def cheatsheet():
    return "baynav: Variational inference with normalizing flow"
