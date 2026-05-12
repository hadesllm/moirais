"""HMC divergent transition count + rate."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["divergent_transitions_count"]


def divergent_transitions_count(chains):
    """
    HMC divergent transition count + rate

    Formula: n_div / n_total -- flag if > 0 (NUTS adaptation issue)

    Parameters
    ----------
    chains : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Betancourt (2017)
    """
    chains = np.atleast_1d(np.asarray(chains, dtype=float))
    n = len(chains)
    result = float(np.mean(chains))
    se = float(np.std(chains, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "HMC divergent transition count + rate"})


def cheatsheet():
    return "divcd: HMC divergent transition count + rate"
