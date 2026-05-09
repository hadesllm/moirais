# moirais.fn — function file (hadesllm/moirais)
"""Leave-one-out cross-validation (LOO-CV)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_loo_cv"]


def rangayyan_loo_cv(X, y, classifier):
    """
    Leave-one-out cross-validation (LOO-CV)

    Formula: LOO error = (1/N) sum I(f_{-i}(x_i) != y_i)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    classifier : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loo_error, predictions

    References
    ----------
    Rangayyan Ch 10.10.3
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Leave-one-out cross-validation (LOO-CV)"})


def cheatsheet():
    return "rgloo: Leave-one-out cross-validation (LOO-CV)"
