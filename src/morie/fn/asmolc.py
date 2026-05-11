"""OLC long-read assembly."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["olc_assembly"]


def olc_assembly(long_reads):
    """
    OLC long-read assembly

    Formula: overlap-layout-consensus on long reads

    Parameters
    ----------
    long_reads : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Myers (2005)
    """
    long_reads = np.atleast_1d(np.asarray(long_reads, dtype=float))
    n = len(long_reads)
    result = float(np.mean(long_reads))
    se = float(np.std(long_reads, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "OLC long-read assembly"})


def cheatsheet():
    return "asmolc: OLC long-read assembly"
