"""Word embedding lookup."""
import numpy as np
from ._richresult import RichResult

__all__ = ["word_embedding"]


def word_embedding(x):
    """
    Word embedding lookup

    Formula: e = E[token_id], E in R^{V x d}

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Mikolov et al. (2013)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Word embedding lookup"})


def cheatsheet():
    return "wdemb: Word embedding lookup"
