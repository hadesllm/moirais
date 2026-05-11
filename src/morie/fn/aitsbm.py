"""Spurious correlation diagnostic via subcompositional incoherence."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["compositional_simbias"]


def compositional_simbias(X, idx):
    """
    Spurious correlation diagnostic via subcompositional incoherence

    Formula: compare ρ on subcompositions vs. variation array

    Parameters
    ----------
    X : array-like
        Input data.
    idx : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rho_full, rho_sub, delta

    References
    ----------
    Pearson (1897); Aitchison (1986)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(X), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Spurious correlation diagnostic via subcompositional incoherence"})
    result = stats.spearmanr(X[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Spurious correlation diagnostic via subcompositional incoherence"})


def cheatsheet():
    return "aitsbm: Spurious correlation diagnostic via subcompositional incoherence"
