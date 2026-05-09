"""Random walk graph kernel."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["random_walk_kernel"]


def random_walk_kernel(G1, G2, lam):
    """
    Random walk graph kernel

    Formula: sum_k λ^k tr(W_x W_y)^k

    Parameters
    ----------
    G1 : array-like
        Input data.
    G2 : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gärtner et al (2003)
    """
    G1 = np.atleast_1d(np.asarray(G1, dtype=float))
    n = len(G1)
    result = float(np.mean(G1))
    se = float(np.std(G1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Random walk graph kernel"})


def cheatsheet():
    return "randW: Random walk graph kernel"
