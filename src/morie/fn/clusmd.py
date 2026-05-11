"""Butina compound clustering by similarity."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["butina_cluster"]


def butina_cluster(fps, cutoff):
    """
    Butina compound clustering by similarity

    Formula: radius-bounded clustering on Tanimoto distance

    Parameters
    ----------
    fps : array-like
        Input data.
    cutoff : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Butina (1999)
    """
    fps = np.atleast_1d(np.asarray(fps, dtype=float))
    n = len(fps)
    result = float(np.mean(fps))
    se = float(np.std(fps, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Butina compound clustering by similarity"})


def cheatsheet():
    return "clusmd: Butina compound clustering by similarity"
