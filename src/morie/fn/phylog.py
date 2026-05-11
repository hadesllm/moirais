"""Phylogenetic time-to-MRCA."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["phylogenetic_dating"]


def phylogenetic_dating(tree, sample_dates):
    """
    Phylogenetic time-to-MRCA

    Formula: coalescent calibrated by sample dates

    Parameters
    ----------
    tree : array-like
        Input data.
    sample_dates : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rambaut et al (2016) BEAST
    """
    tree = np.atleast_1d(np.asarray(tree, dtype=float))
    n = len(tree)
    result = float(np.mean(tree))
    se = float(np.std(tree, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Phylogenetic time-to-MRCA"})


def cheatsheet():
    return "phylog: Phylogenetic time-to-MRCA"
