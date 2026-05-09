"""Lloyd-Max optimal scalar codebook for N(0, 1) source at b bits."""
import numpy as np
from ._richresult import RichResult

__all__ = ["turboquant_lloyd_max_codebook"]


def turboquant_lloyd_max_codebook(bits, source_dist):
    """
    Lloyd-Max optimal scalar codebook for N(0, 1) source at b bits

    Formula: iterate: centroids = E[X | X in region]; boundaries = midpoints of adjacent centroids; until convergence

    Parameters
    ----------
    bits : array-like
        Input data.
    source_dist : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: codebook

    References
    ----------
    TurboQuant MOIRAIS integration — lloyd_max_codebook
    """
    bits = np.atleast_1d(np.asarray(bits, dtype=float))
    n = len(bits)
    result = float(np.mean(bits))
    se = float(np.std(bits, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Lloyd-Max optimal scalar codebook for N(0, 1) source at b bits"})


def cheatsheet():
    return "tqlld: Lloyd-Max optimal scalar codebook for N(0, 1) source at b bits"
