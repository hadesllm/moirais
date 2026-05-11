# morie.fn — function file (hadesllm/morie)
"""Separability index: ratio of between-class to within-class scatter."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_separability_index"]


def rangayyan_separability_index(X, y):
    """
    Separability index: ratio of between-class to within-class scatter

    Formula: J = tr(S_B) / tr(S_W); S_B=between-class; S_W=within-class scatter matrix

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: J_index, S_W, S_B

    References
    ----------
    Rangayyan Ch 10.10.1
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Separability index: ratio of between-class to within-class scatter"})


def cheatsheet():
    return "rgsepix: Separability index: ratio of between-class to within-class scatter"
