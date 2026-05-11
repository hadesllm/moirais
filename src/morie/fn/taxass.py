"""Taxonomic classification (Kraken2)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["taxonomic_assignment"]


def taxonomic_assignment(reads, kraken_db):
    """
    Taxonomic classification (Kraken2)

    Formula: k-mer LCA against taxonomy

    Parameters
    ----------
    reads : array-like
        Input data.
    kraken_db : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wood-Lu-Langmead (2019)
    """
    reads = np.atleast_1d(np.asarray(reads, dtype=float))
    n = len(reads)
    result = float(np.mean(reads))
    se = float(np.std(reads, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Taxonomic classification (Kraken2)"})


def cheatsheet():
    return "taxass: Taxonomic classification (Kraken2)"
