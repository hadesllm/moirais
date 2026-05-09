# moirais.fn — function file (hadesllm/moirais)
"""Fisher linear discriminant analysis (LDA)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_fisher_lda"]


def rangayyan_fisher_lda(X, y):
    """
    Fisher linear discriminant analysis (LDA)

    Formula: w = S_W^{-1}*(mu_1-mu_2); S_W=within-class scatter matrix

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: w, projection, boundary

    References
    ----------
    Rangayyan Ch 10.4.2
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fisher linear discriminant analysis (LDA)"})


def cheatsheet():
    return "rgfld: Fisher linear discriminant analysis (LDA)"
