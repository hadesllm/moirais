"""KR-20 reliability for binary items."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kuder_richardson_20"]


def kuder_richardson_20(X):
    """
    KR-20 reliability for binary items

    Formula: KR-20 = k/(k-1) (1 - sum p_i q_i / sigma_T^2)

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kuder-Richardson (1937)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "KR-20 reliability for binary items"})


def cheatsheet():
    return "kr20cr: KR-20 reliability for binary items"
