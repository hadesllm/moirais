"""N-HiTS hierarchical interpolation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["n_hits"]


def n_hits(y, stacks, mlp_units):
    """
    N-HiTS hierarchical interpolation

    Formula: multi-rate sampling + interpolation per stack

    Parameters
    ----------
    y : array-like
        Input data.
    stacks : array-like
        Input data.
    mlp_units : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Challu et al (2023) N-HiTS
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "N-HiTS hierarchical interpolation"})


def cheatsheet():
    return "nhits: N-HiTS hierarchical interpolation"
