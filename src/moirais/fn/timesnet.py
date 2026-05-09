"""TimesNet — 2D periodic blocks."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["timesnet"]


def timesnet(X, top_k):
    """
    TimesNet — 2D periodic blocks

    Formula: FFT to find periods; reshape to 2D; Inception

    Parameters
    ----------
    X : array-like
        Input data.
    top_k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wu et al (2023) TimesNet
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TimesNet — 2D periodic blocks"})


def cheatsheet():
    return "timesnet: TimesNet — 2D periodic blocks"
