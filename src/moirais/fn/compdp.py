"""Basic DP composition."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["basic_composition"]


def basic_composition(epsilons):
    """
    Basic DP composition

    Formula: k ε-DP mechs are kε-DP

    Parameters
    ----------
    epsilons : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dwork-McSherry-Nissim-Smith (2006)
    """
    epsilons = np.atleast_1d(np.asarray(epsilons, dtype=float))
    n = len(epsilons)
    result = float(np.mean(epsilons))
    se = float(np.std(epsilons, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Basic DP composition"})


def cheatsheet():
    return "compdp: Basic DP composition"
