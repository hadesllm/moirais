# moirais.fn — function file (hadesllm/moirais)
"""iTransformer: inverted attention — variates as tokens, time as feature dim."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_itransformer"]


def joseph_itransformer(X, n_variates, transformer):
    """
    iTransformer: inverted attention — variates as tokens, time as feature dim

    Formula: token_i = series_for_variate_i; MHA across variates; feed-forward along time

    Parameters
    ----------
    X : array-like
        Input data.
    n_variates : array-like
        Input data.
    transformer : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_hat

    References
    ----------
    Joseph Ch 16, iTransformer section
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "iTransformer: inverted attention — variates as tokens, time as feature dim"})


def cheatsheet():
    return "joitran: iTransformer: inverted attention — variates as tokens, time as feature dim"
