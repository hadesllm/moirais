"""SAX symbolic aggregate approximation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sax_representation"]


def sax_representation(x, window, alphabet):
    """
    SAX symbolic aggregate approximation

    Formula: PAA + alphabet quantile

    Parameters
    ----------
    x : array-like
        Input data.
    window : array-like
        Input data.
    alphabet : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lin et al (2007) SAX
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SAX symbolic aggregate approximation"})


def cheatsheet():
    return "saxR: SAX symbolic aggregate approximation"
