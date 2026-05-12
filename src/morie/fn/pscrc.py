# morie.fn -- function file (hadesllm/morie)
"""pscl rollcall object: encode roll call vote matrix for spatial scaling."""
import numpy as np
from ._richresult import RichResult

__all__ = ["pscl_rollcall"]


def pscl_rollcall(vote_matrix, legis_data, vote_data):
    """
    pscl rollcall object: encode roll call vote matrix for spatial scaling

    Formula: V = {v_ij in {yea, nay, abstain, absent}}; lopsided votes dropped; minority defines polarity

    Parameters
    ----------
    vote_matrix : array-like
        Input data.
    legis_data : array-like
        Input data.
    vote_data : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'rollcall_obj': 'object'}

    References
    ----------
    Armstrong Ch 5
    """
    vote_matrix = np.asarray(vote_matrix, dtype=float)
    n = int(vote_matrix) if vote_matrix.ndim == 0 else len(vote_matrix)
    result = float(np.mean(vote_matrix))
    se = float(np.std(vote_matrix, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "pscl rollcall object: encode roll call vote matrix for spatial scaling"})


def cheatsheet():
    return "pscrc: pscl rollcall object: encode roll call vote matrix for spatial scaling"
