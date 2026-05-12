# morie.fn -- function file (hadesllm/morie)
"""Rice index of party cohesion on roll calls."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rice_index"]


def rice_index(votes, party_id):
    """
    Rice index of party cohesion on roll calls

    Formula: Rice_j = |%yea_party_j - %nay_party_j| in [0,1]

    Parameters
    ----------
    votes : array-like
        Input data.
    party_id : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'rice': 'array'}

    References
    ----------
    Armstrong Ch 5
    """
    votes = np.asarray(votes, dtype=float)
    n = int(votes) if votes.ndim == 0 else len(votes)
    result = float(np.mean(votes))
    se = float(np.std(votes, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rice index of party cohesion on roll calls"})


def cheatsheet():
    return "ricei: Rice index of party cohesion on roll calls"
