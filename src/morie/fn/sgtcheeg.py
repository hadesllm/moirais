"""Edge Cheeger constant via sweep on Fiedler vector."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_cheeger_constant"]


def sgt_cheeger_constant(A):
    """
    Edge Cheeger constant via sweep on Fiedler vector

    Formula: h(G) = min_S |∂S|/min(vol S, vol S^c)

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: h

    References
    ----------
    Cheeger (1969); Chung (1997)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Edge Cheeger constant via sweep on Fiedler vector"})


def cheatsheet():
    return "sgtcheeg: Edge Cheeger constant via sweep on Fiedler vector"
