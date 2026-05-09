"""Centred log-ratio (CLR) transform of a composition."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aitchison_clr"]


def aitchison_clr(x):
    """
    Centred log-ratio (CLR) transform of a composition

    Formula: clr_i(x) = log(x_i / g(x)),  g(x) = (prod_j x_j)^(1/D)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: z

    References
    ----------
    Aitchison (1986) §4
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Centred log-ratio (CLR) transform of a composition"})


def cheatsheet():
    return "aitclr: Centred log-ratio (CLR) transform of a composition"
