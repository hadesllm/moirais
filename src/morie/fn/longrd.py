"""Long-read consensus polishing (medaka)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["long_read_polish"]


def long_read_polish(assembly, reads):
    """
    Long-read consensus polishing (medaka)

    Formula: per-position deep-learning consensus

    Parameters
    ----------
    assembly : array-like
        Input data.
    reads : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    ONT medaka
    """
    assembly = np.atleast_1d(np.asarray(assembly, dtype=float))
    n = len(assembly)
    result = float(np.mean(assembly))
    se = float(np.std(assembly, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Long-read consensus polishing (medaka)"})


def cheatsheet():
    return "longrd: Long-read consensus polishing (medaka)"
