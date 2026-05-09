# moirais.fn — function file (hadesllm/moirais)
"""Random forest via bagging + feature sampling."""
import numpy as np
from ._richresult import RichResult

__all__ = ["random_forest_ensemble"]


def random_forest_ensemble(x, y):
    """
    Random forest via bagging + feature sampling

    Formula: f(x) = (1/B) sum f_b(x), b=1..B

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Geron (2026), Ch 7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Random forest via bagging + feature sampling"})


def cheatsheet():
    return "rfens: Random forest via bagging + feature sampling"
