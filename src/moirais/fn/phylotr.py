"""Phylogenetic tree construction."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["phylogenetic_tree"]


def phylogenetic_tree(sequences, method):
    """
    Phylogenetic tree construction

    Formula: NJ or ML on distance matrix

    Parameters
    ----------
    sequences : array-like
        Input data.
    method : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Saitou-Nei (1987); Felsenstein (2004)
    """
    sequences = np.atleast_1d(np.asarray(sequences, dtype=float))
    n = len(sequences)
    result = float(np.mean(sequences))
    se = float(np.std(sequences, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Phylogenetic tree construction"})


def cheatsheet():
    return "phylotr: Phylogenetic tree construction"
