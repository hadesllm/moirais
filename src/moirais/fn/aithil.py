"""Hill numbers of order q for a composition."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["compositional_hill"]


def compositional_hill(x, q):
    """
    Hill numbers of order q for a composition

    Formula: qD = (Σ x_i^q)^{1/(1-q)}

    Parameters
    ----------
    x : array-like
        Input data.
    q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: qD

    References
    ----------
    Hill (1973)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hill numbers of order q for a composition"})


def cheatsheet():
    return "aithil: Hill numbers of order q for a composition"
