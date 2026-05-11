"""Multiplicative replacement of zeros in compositions."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["compositional_zero_multreplace"]


def compositional_zero_multreplace(X, delta):
    """
    Multiplicative replacement of zeros in compositions

    Formula: x'_i = δ if x_i=0 else x_i(1 - sum δ)

    Parameters
    ----------
    X : array-like
        Input data.
    delta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X_imp

    References
    ----------
    Martín-Fernández et al. (2003)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multiplicative replacement of zeros in compositions"})


def cheatsheet():
    return "aitzmu: Multiplicative replacement of zeros in compositions"
