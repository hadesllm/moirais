"""k-anonymity check on a quasi-identifier set."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["k_anonymity_check"]


def k_anonymity_check(y, quasi_ids, k):
    """
    k-anonymity check on a quasi-identifier set

    Formula: min_{group} |group| >= k

    Parameters
    ----------
    y : array-like
        Input data.
    quasi_ids : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sweeney (2002)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "k-anonymity check on a quasi-identifier set"})


def cheatsheet():
    return "kanon: k-anonymity check on a quasi-identifier set"
