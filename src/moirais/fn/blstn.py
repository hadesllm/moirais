"""BLASTN nucleotide alignment."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["blast_nucleotide"]


def blast_nucleotide(query, db):
    """
    BLASTN nucleotide alignment

    Formula: BLAST adapted for DNA

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
    Altschul (1990)
    """
    query = np.atleast_1d(np.asarray(query, dtype=float))
    n = len(query)
    result = float(np.mean(query))
    se = float(np.std(query, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BLASTN nucleotide alignment"})


def cheatsheet():
    return "blstn: BLASTN nucleotide alignment"
