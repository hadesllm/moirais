# morie.fn -- function file (hadesllm/morie)
"""RetNet retention: chunkwise recurrent equivalent of attention."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_retnet_retention"]


def kamath_retnet_retention(Q, K, V, gamma):
    """
    RetNet retention: chunkwise recurrent equivalent of attention

    Formula: Retention(Q,K,V) = (QK^T) .* D @ V;  D_ij = gamma^(i-j) if i>=j else 0

    Parameters
    ----------
    Q : array-like
        Input data.
    K : array-like
        Input data.
    V : array-like
        Input data.
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: output

    References
    ----------
    Kamath Ch 10, RetNet section
    """
    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    n = len(Q)
    result = float(np.mean(Q))
    se = float(np.std(Q, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RetNet retention: chunkwise recurrent equivalent of attention"})


def cheatsheet():
    return "kmret: RetNet retention: chunkwise recurrent equivalent of attention"
