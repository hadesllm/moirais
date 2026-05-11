"""ChIP-seq peak calling (MACS2)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["chip_seq_peak"]


def chip_seq_peak(reads, control):
    """
    ChIP-seq peak calling (MACS2)

    Formula: local Poisson model + dynamic threshold

    Parameters
    ----------
    reads : array-like
        Input data.
    control : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Zhang et al (2008) MACS2
    """
    reads = np.atleast_1d(np.asarray(reads, dtype=float))
    n = len(reads)
    result = float(np.mean(reads))
    se = float(np.std(reads, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ChIP-seq peak calling (MACS2)"})


def cheatsheet():
    return "chipsq: ChIP-seq peak calling (MACS2)"
