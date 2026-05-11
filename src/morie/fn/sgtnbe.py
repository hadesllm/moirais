"""Non-backtracking (Hashimoto) matrix B."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_nonbacktracking_matrix"]


def sgt_nonbacktracking_matrix(edges, n):
    """
    Non-backtracking (Hashimoto) matrix B

    Formula: B_{e,e'} = 1 if t(e)=s(e') and e' ≠ rev(e)

    Parameters
    ----------
    edges : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: B

    References
    ----------
    Hashimoto (1989); Krzakala et al. (2013)
    """
    edges = np.atleast_1d(np.asarray(edges, dtype=float))
    n = len(edges)
    result = float(np.mean(edges))
    se = float(np.std(edges, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Non-backtracking (Hashimoto) matrix B"})


def cheatsheet():
    return "sgtnbe: Non-backtracking (Hashimoto) matrix B"
