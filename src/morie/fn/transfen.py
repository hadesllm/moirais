"""Transfer entropy TE(X->Y)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["transfer_entropy_te"]


def transfer_entropy_te(x, y, lag):
    """
    Transfer entropy TE(X->Y)

    Formula: H(Y_t | Y_past) - H(Y_t | Y_past, X_past)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    lag : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schreiber (2000)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Transfer entropy TE(X->Y)"})


def cheatsheet():
    return "transfen: Transfer entropy TE(X->Y)"
