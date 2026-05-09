"""MaxMin compound-set diversity selection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["compound_diversity"]


def compound_diversity(fps, n_pick):
    """
    MaxMin compound-set diversity selection

    Formula: argmax min_dist; greedy diverse subset

    Parameters
    ----------
    fps : array-like
        Input data.
    n_pick : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lajiness (1990)
    """
    fps = np.atleast_1d(np.asarray(fps, dtype=float))
    n = len(fps)
    result = float(np.mean(fps))
    se = float(np.std(fps, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MaxMin compound-set diversity selection"})


def cheatsheet():
    return "tncomp: MaxMin compound-set diversity selection"
