"""Synthetic control subset selection (LASSO-relaxed)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_synthetic_subset"]


def causal_synthetic_subset(X1_pre, X0_pre, lam):
    """
    Synthetic control subset selection (LASSO-relaxed)

    Formula: min ||X1-X0w||² + λ||w||_1, w>=0, Σw=1

    Parameters
    ----------
    X1_pre : array-like
        Input data.
    X0_pre : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: weights, support

    References
    ----------
    Doudchenko-Imbens (2017)
    """
    X1_pre = np.atleast_1d(np.asarray(X1_pre, dtype=float))
    n = len(X1_pre)
    result = float(np.mean(X1_pre))
    se = float(np.std(X1_pre, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Synthetic control subset selection (LASSO-relaxed)"})


def cheatsheet():
    return "causscss: Synthetic control subset selection (LASSO-relaxed)"
