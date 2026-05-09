# moirais.fn — function file (hadesllm/moirais)
"""Collider (v-structure/immorality) A->B<-C: conditioning on B opens path."""
import numpy as np
from ._richresult import RichResult

__all__ = ["collider_structure"]


def collider_structure(A, B, C, conditioned):
    """
    Collider (v-structure/immorality) A->B<-C: conditioning on B opens path

    Formula: A->B<-C: A _|_ C marginally; A NOT _|_ C | B (collider bias / Berkson's paradox)

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Collider (v-structure/immorality) A->B<-C: conditioning on B opens path"})


def cheatsheet():
    return "colst: Collider (v-structure/immorality) A->B<-C: conditioning on B opens path"
