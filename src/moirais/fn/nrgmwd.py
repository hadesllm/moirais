"""Normalized random measure."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["normalized_random_measure"]


def normalized_random_measure(y, alpha, tau):
    """
    Normalized random measure

    Formula: NRM = mu_alpha / mu_alpha(R)

    Parameters
    ----------
    y : array-like
        Input data.
    alpha : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Regazzini-Lijoi-Prünster (2003)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Normalized random measure"})


def cheatsheet():
    return "nrgmwd: Normalized random measure"
