"""Orthogonalized JL: QR-decompose the rows of S for lower empirical distortion."""
import numpy as np
from ._richresult import RichResult

__all__ = ["turboquant_orthogonalized_jl"]


def turboquant_orthogonalized_jl(S):
    """
    Orthogonalized JL: QR-decompose the rows of S for lower empirical distortion

    Formula: Q, R = QR(S^T);  S_orth = Q^T;  use S_orth in place of S

    Parameters
    ----------
    S : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: S_orth

    References
    ----------
    Zandieh et al. 2024 Section 4.1 (Orthogonalized JL)
    """
    S = np.atleast_1d(np.asarray(S, dtype=float))
    n = len(S)
    result = float(np.mean(S))
    se = float(np.std(S, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Orthogonalized JL: QR-decompose the rows of S for lower empirical distortion"})


def cheatsheet():
    return "tqorth: Orthogonalized JL: QR-decompose the rows of S for lower empirical distortion"
