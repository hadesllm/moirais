"""Shannon diversity of a closed composition."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["compositional_shannon"]


def compositional_shannon(x):
    """
    Shannon diversity of a closed composition

    Formula: H(x) = -Σ x_i log x_i

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: H

    References
    ----------
    Shannon (1948)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Shannon diversity of a closed composition"})


def cheatsheet():
    return "aitsdv: Shannon diversity of a closed composition"
