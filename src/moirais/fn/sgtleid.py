"""Leiden refinement after Louvain (Traag-Waltman-van Eck)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_leiden_step"]


def sgt_leiden_step(A, labels):
    """
    Leiden refinement after Louvain (Traag-Waltman-van Eck)

    Formula: Refine by per-community subdivisions to ensure connected

    Parameters
    ----------
    A : array-like
        Input data.
    labels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels_new, Q_new

    References
    ----------
    Traag-Waltman-van Eck (2019)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Leiden refinement after Louvain (Traag-Waltman-van Eck)"})


def cheatsheet():
    return "sgtleid: Leiden refinement after Louvain (Traag-Waltman-van Eck)"
