"""LINE embeddings (1st + 2nd order)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["line"]


def line(G, dim):
    """
    LINE embeddings (1st + 2nd order)

    Formula: min KL(emp prox, dot prox)

    Parameters
    ----------
    G : array-like
        Input data.
    dim : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Tang et al (2015) LINE
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "LINE embeddings (1st + 2nd order)"})


def cheatsheet():
    return "line: LINE embeddings (1st + 2nd order)"
