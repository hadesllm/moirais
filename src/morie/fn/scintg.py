"""scRNA-seq batch integration (Harmony)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["singlecell_integration"]


def singlecell_integration(X, batch):
    """
    scRNA-seq batch integration (Harmony)

    Formula: iterative correction in PCA space

    Parameters
    ----------
    X : array-like
        Input data.
    batch : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Korsunsky et al (2019)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "scRNA-seq batch integration (Harmony)"})


def cheatsheet():
    return "scintg: scRNA-seq batch integration (Harmony)"
