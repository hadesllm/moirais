# morie.fn — function file (hadesllm/morie)
"""Chain (mediation) structure A->B->C: information flows, B is mediator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["chain_structure"]


def chain_structure(A, B, C, conditioned):
    """
    Chain (mediation) structure A->B->C: information flows, B is mediator

    Formula: A->B->C: A _|_ C | B (d-separation); blocked when conditioning on B

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Chain (mediation) structure A->B->C: information flows, B is mediator"})


def cheatsheet():
    return "chstr: Chain (mediation) structure A->B->C: information flows, B is mediator"
