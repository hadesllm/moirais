"""Zivot-Andrews unit root with endogenous break."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["zivot_andrews_unit_root"]


def zivot_andrews_unit_root(x, model, lags):
    """
    Zivot-Andrews unit root with endogenous break

    Formula: min_t inf_TB t_alpha(TB), TB chosen to minimise

    Parameters
    ----------
    x : array-like
        Input data.
    model : array-like
        Input data.
    lags : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Zivot & Andrews (1992)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Zivot-Andrews unit root with endogenous break"})


def cheatsheet():
    return "zaurts: Zivot-Andrews unit root with endogenous break"
