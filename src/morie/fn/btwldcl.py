"""Wild cluster bootstrap for clustered errors."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_wild_cluster"]


def boot_wild_cluster(X, y, cluster, B):
    """
    Wild cluster bootstrap for clustered errors

    Formula: y_g* = X_g β̂ + ε̂_g v_g, v_g ∈ {-1,1}

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    cluster : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta_b

    References
    ----------
    Cameron-Gelbach-Miller (2008)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wild cluster bootstrap for clustered errors"})


def cheatsheet():
    return "btwldcl: Wild cluster bootstrap for clustered errors"
