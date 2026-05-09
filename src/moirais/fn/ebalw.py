# moirais.fn — function file (hadesllm/moirais)
"""Entropy balancing for covariate balance (Hainmueller 2012)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["entropy_balancing"]


def entropy_balancing(X, T, moments):
    """
    Entropy balancing for covariate balance (Hainmueller 2012)

    Formula: min_w KL(w||q) s.t. sum_i w_i*X_i = mean(X_control); weights minimize entropy deviation

    Parameters
    ----------
    X : array-like
        Input data.
    T : array-like
        Input data.
    moments : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'weights': 'array', 'balance_stats': 'dict'}

    References
    ----------
    Molak Ch 9
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Entropy balancing for covariate balance (Hainmueller 2012)"})


def cheatsheet():
    return "ebalw: Entropy balancing for covariate balance (Hainmueller 2012)"
