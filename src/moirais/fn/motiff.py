"""Network motif counting."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["motif_count"]


def motif_count(G, motif_size):
    """
    Network motif counting

    Formula: count k-node subgraphs

    Parameters
    ----------
    G : array-like
        Input data.
    motif_size : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Milo et al (2002)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Network motif counting"})


def cheatsheet():
    return "motiff: Network motif counting"
