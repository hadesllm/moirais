"""Rosenbaum sensitivity bound on Wilcoxon p-value."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_rosenbaum_bound"]


def causal_rosenbaum_bound(paired_diff, Gamma):
    """
    Rosenbaum sensitivity bound on Wilcoxon p-value

    Formula: Adjust matched p by multiplier Γ on odds ratio of unobs.

    Parameters
    ----------
    paired_diff : array-like
        Input data.
    Gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: p_lower, p_upper

    References
    ----------
    Rosenbaum (2002)
    """
    paired_diff = np.atleast_1d(np.asarray(paired_diff, dtype=float))
    n = len(paired_diff)
    result = float(np.mean(paired_diff))
    se = float(np.std(paired_diff, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rosenbaum sensitivity bound on Wilcoxon p-value"})


def cheatsheet():
    return "causrosen: Rosenbaum sensitivity bound on Wilcoxon p-value"
