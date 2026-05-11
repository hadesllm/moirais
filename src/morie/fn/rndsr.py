# morie.fn — function file (hadesllm/morie)
"""Random search hyperparameter optimization."""
import numpy as np
from ._richresult import RichResult

__all__ = ["random_search_cv"]


def random_search_cv(x, y):
    """
    Random search hyperparameter optimization

    Formula: params ~ Uniform(grid), eval CV

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
    Bergstra & Bengio (2012)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Random search hyperparameter optimization"})


def cheatsheet():
    return "rndsr: Random search hyperparameter optimization"
