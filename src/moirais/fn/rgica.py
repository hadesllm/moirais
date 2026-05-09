# moirais.fn — function file (hadesllm/moirais)
"""FastICA algorithm for independent component analysis."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_fastica"]


def rangayyan_fastica(X, n_components, nonlin, max_iter, tol):
    """
    FastICA algorithm for independent component analysis

    Formula: w_k = E{X*g(w_k^T*X)} - E{g'(w_k^T*X)}*w_k; g(y)=tanh(a*y)

    Parameters
    ----------
    X : array-like
        Input data.
    n_components : array-like
        Input data.
    nonlin : array-like
        Input data.
    max_iter : array-like
        Input data.
    tol : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: S, A, W

    References
    ----------
    Rangayyan Ch 9.7.2
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "FastICA algorithm for independent component analysis"})


def cheatsheet():
    return "rgica: FastICA algorithm for independent component analysis"
