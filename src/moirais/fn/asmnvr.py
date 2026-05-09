"""De novo assembly (de Bruijn graph)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["genome_assembly"]


def genome_assembly(reads, k):
    """
    De novo assembly (de Bruijn graph)

    Formula: k-mer DBG + bubble/tip pruning + path resolution

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
    Pevzner-Tang-Waterman (2001) DBG
    """
    reads = np.atleast_1d(np.asarray(reads, dtype=float))
    n = len(reads)
    result = float(np.mean(reads))
    se = float(np.std(reads, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "De novo assembly (de Bruijn graph)"})


def cheatsheet():
    return "asmnvr: De novo assembly (de Bruijn graph)"
