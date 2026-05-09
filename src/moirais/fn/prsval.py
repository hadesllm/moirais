"""Messick validity argument."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["presmessick_validity"]


def presmessick_validity(evidence_set):
    """
    Messick validity argument

    Formula: unified concept: content + construct + criterion

    Parameters
    ----------
    evidence_set : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Messick (1989)
    """
    evidence_set = np.atleast_1d(np.asarray(evidence_set, dtype=float))
    n = len(evidence_set)
    result = float(np.mean(evidence_set))
    se = float(np.std(evidence_set, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Messick validity argument"})


def cheatsheet():
    return "prsval: Messick validity argument"
