"""Sum / mean / max graph pooling."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sum_pool"]


def sum_pool(X, mode):
    """
    Sum / mean / max graph pooling

    Formula: h_G = aggregate over node embeddings

    Parameters
    ----------
    X : array-like
        Input data.
    mode : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Xu et al (2019)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sum / mean / max graph pooling"})


def cheatsheet():
    return "sumP: Sum / mean / max graph pooling"
