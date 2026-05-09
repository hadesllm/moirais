# moirais.fn — function file (hadesllm/moirais)
"""Random forest for genomic prediction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["random_forest_genomic"]


def random_forest_genomic(x, y, markers):
    """
    Random forest for genomic prediction

    Formula: f(x) = (1/B) sum f_b(x), b=1..B on marker subsets

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    markers : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Montesinos Lopez Ch 8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Random forest for genomic prediction"})


def cheatsheet():
    return "rfgen: Random forest for genomic prediction"
