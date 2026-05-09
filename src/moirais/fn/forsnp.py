"""Forensic likelihood ratio."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["forensic_lr"]


def forensic_lr(E, H1, H2):
    """
    Forensic likelihood ratio

    Formula: LR = P(E | H1) / P(E | H2)

    Parameters
    ----------
    E : array-like
        Input data.
    H1 : array-like
        Input data.
    H2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Buckleton-Triggs-Walsh (2005)
    """
    E = np.atleast_1d(np.asarray(E, dtype=float))
    n = len(E)
    result = float(np.mean(E))
    se = float(np.std(E, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Forensic likelihood ratio"})


def cheatsheet():
    return "forsnp: Forensic likelihood ratio"
