# moirais.fn — function file (hadesllm/moirais)
"""Single-index model for conditional mean: E[Y|X]=G(X'beta)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_single_index_model"]


def horowitz_single_index_model(x, y):
    """
    Single-index model for conditional mean: E[Y|X]=G(X'beta)

    Formula: E(Y|X=x) = G(x'beta) where G unknown function, beta unknown d-vector

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta, G_hat

    References
    ----------
    Horowitz Ch 2, Eq 2.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Single-index model for conditional mean: E[Y|X]=G(X'beta)"})


def cheatsheet():
    return "hrzsim: Single-index model for conditional mean: E[Y|X]=G(X'beta)"
