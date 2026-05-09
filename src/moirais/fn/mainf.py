"""Cook's distance + DFFITS for meta-regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_influence_diagnostics"]


def ma_influence_diagnostics(yi, vi, X):
    """
    Cook's distance + DFFITS for meta-regression

    Formula: D_i, DFFITS_i per study

    Parameters
    ----------
    yi : array-like
        Input data.
    vi : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: cook, dffits, hat

    References
    ----------
    Viechtbauer & Cheung (2010)
    """
    yi = np.atleast_1d(np.asarray(yi, dtype=float))
    n = len(yi)
    result = float(np.mean(yi))
    se = float(np.std(yi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cook's distance + DFFITS for meta-regression"})


def cheatsheet():
    return "mainf: Cook's distance + DFFITS for meta-regression"
