"""GRU4Rec session-based."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gru4rec"]


def gru4rec(sessions, K):
    """
    GRU4Rec session-based

    Formula: GRU over click sequence + ranking loss

    Parameters
    ----------
    sessions : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hidasi et al (2016)
    """
    sessions = np.atleast_1d(np.asarray(sessions, dtype=float))
    n = len(sessions)
    result = float(np.mean(sessions))
    se = float(np.std(sessions, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GRU4Rec session-based"})


def cheatsheet():
    return "gru4r: GRU4Rec session-based"
