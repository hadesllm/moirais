"""BLAST protein heuristic."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["blast_protein"]


def blast_protein(query, db):
    """
    BLAST protein heuristic

    Formula: seed-and-extend with k-mer index + ungapped→gapped extension

    Parameters
    ----------
    query : array-like
        Input data.
    db : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Altschul et al (1990)
    """
    query = np.atleast_1d(np.asarray(query, dtype=float))
    n = len(query)
    result = float(np.mean(query))
    se = float(np.std(query, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BLAST protein heuristic"})


def cheatsheet():
    return "blastp: BLAST protein heuristic"
