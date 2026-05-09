# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Party unity score per legislator: how often votes with party majority."""
import numpy as np
from ._richresult import RichResult

__all__ = ["party_unity_score"]


def party_unity_score(vote_matrix, party_id):
    """
    Party unity score per legislator: how often votes with party majority

    Formula: Unity_i = #{v: i votes with party majority} / #{v: i voted}

    Parameters
    ----------
    vote_matrix : array-like
        Input data.
    party_id : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'unity': 'array'}

    References
    ----------
    Armstrong Ch 5
    """
    vote_matrix = np.asarray(vote_matrix, dtype=float)
    n = int(vote_matrix) if vote_matrix.ndim == 0 else len(vote_matrix)
    result = float(np.mean(vote_matrix))
    se = float(np.std(vote_matrix, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Party unity score per legislator: how often votes with party majority"})


def cheatsheet():
    return "agpar: Party unity score per legislator: how often votes with party majority"
