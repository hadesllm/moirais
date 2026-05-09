# moirais.fn — function file (hadesllm/moirais)
"""FactScore: fraction of atomic claims in a generation that are factually supported."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_factscore"]


def kamath_factscore(atomic_claims, knowledge_base):
    """
    FactScore: fraction of atomic claims in a generation that are factually supported

    Formula: FactScore = |supported_atomic_claims| / |total_atomic_claims|

    Parameters
    ----------
    atomic_claims : array-like
        Input data.
    knowledge_base : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: score

    References
    ----------
    Kamath Ch 6, FactScore section
    """
    atomic_claims = np.atleast_1d(np.asarray(atomic_claims, dtype=float))
    n = len(atomic_claims)
    result = float(np.mean(atomic_claims))
    se = float(np.std(atomic_claims, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "FactScore: fraction of atomic claims in a generation that are factually supported"})


def cheatsheet():
    return "kmfact: FactScore: fraction of atomic claims in a generation that are factually supported"
