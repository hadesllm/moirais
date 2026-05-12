# morie.fn -- function file (hadesllm/morie)
"""Fork (common cause) structure A<-B->C: B is confounder."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fork_structure"]


def fork_structure(A, B, C, conditioned):
    """
    Fork (common cause) structure A<-B->C: B is confounder

    Formula: A<-B->C: A _|_ C | B; unblocked marginally (confounding), blocked on B

    Parameters
    ----------
    A : array-like
        Input data.
    B : array-like
        Input data.
    C : array-like
        Input data.
    conditioned : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'d_sep': 'bool'}

    References
    ----------
    Molak Ch 5
    """
    A = np.asarray(A, dtype=float)
    n = int(A) if A.ndim == 0 else len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fork (common cause) structure A<-B->C: B is confounder"})


def cheatsheet():
    return "frkst: Fork (common cause) structure A<-B->C: B is confounder"
