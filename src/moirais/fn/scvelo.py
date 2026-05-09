"""scVelo RNA velocity."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rna_velocity"]


def rna_velocity(spliced, unspliced):
    """
    scVelo RNA velocity

    Formula: steady-state spliced/unspliced ratio dynamics

    Parameters
    ----------
    spliced : array-like
        Input data.
    unspliced : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bergen et al (2020)
    """
    spliced = np.atleast_1d(np.asarray(spliced, dtype=float))
    n = len(spliced)
    result = float(np.mean(spliced))
    se = float(np.std(spliced, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "scVelo RNA velocity"})


def cheatsheet():
    return "scvelo: scVelo RNA velocity"
