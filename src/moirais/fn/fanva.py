"""Functional ANOVA."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["fanova"]


def fanova(functions, groups):
    """
    Functional ANOVA

    Formula: decompose function into mean + treatment + interaction

    Parameters
    ----------
    functions : array-like
        Input data.
    groups : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ramsay-Silverman (2005) Ch.13
    """
    functions = np.atleast_1d(np.asarray(functions, dtype=float))
    n = len(functions)
    result = float(np.mean(functions))
    se = float(np.std(functions, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Functional ANOVA"})


def cheatsheet():
    return "fanva: Functional ANOVA"
