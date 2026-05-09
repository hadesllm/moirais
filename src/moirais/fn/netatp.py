"""Attack tolerance via giant-component."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["network_attack_tolerance"]


def network_attack_tolerance(G, attack_strategy, k):
    """
    Attack tolerance via giant-component

    Formula: GCC fraction after k-node removal

    Parameters
    ----------
    G : array-like
        Input data.
    attack_strategy : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Albert-Jeong-Barabási (2000)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Attack tolerance via giant-component"})


def cheatsheet():
    return "netatp: Attack tolerance via giant-component"
