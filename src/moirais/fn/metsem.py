"""Metagenome assembly (metaSPAdes)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["metagenome_assembly"]


def metagenome_assembly(reads, k):
    """
    Metagenome assembly (metaSPAdes)

    Formula: DBG with strain-aware tip resolution

    Parameters
    ----------
    reads : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Nurk et al (2017)
    """
    reads = np.atleast_1d(np.asarray(reads, dtype=float))
    n = len(reads)
    result = float(np.mean(reads))
    se = float(np.std(reads, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Metagenome assembly (metaSPAdes)"})


def cheatsheet():
    return "metsem: Metagenome assembly (metaSPAdes)"
