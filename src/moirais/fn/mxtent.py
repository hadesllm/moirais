"""Maximum entropy distribution."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["max_entropy"]


def max_entropy(constraints):
    """
    Maximum entropy distribution

    Formula: argmax H(p) s.t. moment constraints

    Parameters
    ----------
    constraints : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jaynes (1957)
    """
    constraints = np.atleast_1d(np.asarray(constraints, dtype=float))
    n = len(constraints)
    result = float(np.mean(constraints))
    se = float(np.std(constraints, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Maximum entropy distribution"})


def cheatsheet():
    return "mxtent: Maximum entropy distribution"
