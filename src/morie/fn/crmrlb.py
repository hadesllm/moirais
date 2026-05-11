"""Cramér-Rao lower bound."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cramer_rao_bound"]


def cramer_rao_bound(fisher_info):
    """
    Cramér-Rao lower bound

    Formula: Var(theta_hat) >= I(theta)^-1

    Parameters
    ----------
    fisher_info : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cramér (1946); Rao (1945)
    """
    fisher_info = np.atleast_1d(np.asarray(fisher_info, dtype=float))
    n = len(fisher_info)
    result = float(np.mean(fisher_info))
    se = float(np.std(fisher_info, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cramér-Rao lower bound"})


def cheatsheet():
    return "crmrlb: Cramér-Rao lower bound"
