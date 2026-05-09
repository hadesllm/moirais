"""Fano's inequality on error probability."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["fano_inequality"]


def fano_inequality(pe, X_card):
    """
    Fano's inequality on error probability

    Formula: H(X|Y) <= H(P_e) + P_e log(|X|-1)

    Parameters
    ----------
    pe : array-like
        Input data.
    X_card : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fano (1961)
    """
    pe = np.atleast_1d(np.asarray(pe, dtype=float))
    n = len(pe)
    result = float(np.mean(pe))
    se = float(np.std(pe, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fano's inequality on error probability"})


def cheatsheet():
    return "fanocb: Fano's inequality on error probability"
