# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Agreement score matrix from roll call votes."""
import numpy as np
from ._richresult import RichResult

__all__ = ["agreement_score_matrix"]


def agreement_score_matrix(vote_matrix):
    """
    Agreement score matrix from roll call votes

    Formula: A_ij = #{v: vote_iv == vote_jv} / #{v: both i,j voted}; proportion same votes

    Parameters
    ----------
    vote_matrix : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'A': 'matrix'}

    References
    ----------
    Armstrong Ch 3
    """
    vote_matrix = np.asarray(vote_matrix, dtype=float)
    n = int(vote_matrix) if vote_matrix.ndim == 0 else len(vote_matrix)
    result = float(np.mean(vote_matrix))
    se = float(np.std(vote_matrix, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Agreement score matrix from roll call votes"})


def cheatsheet():
    return "agrsc: Agreement score matrix from roll call votes"
