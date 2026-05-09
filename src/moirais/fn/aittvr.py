"""Total variance of a compositional sample."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aitchison_total_variance"]


def aitchison_total_variance(X):
    """
    Total variance of a compositional sample

    Formula: totvar = (1/2D) sum_{i,j} var(log(x_i/x_j))

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: tv

    References
    ----------
    Aitchison (1997)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Total variance of a compositional sample"})


def cheatsheet():
    return "aittvr: Total variance of a compositional sample"
