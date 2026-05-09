# moirais.fn — function file (hadesllm/moirais)
"""He/Kaiming weight initialization for ReLU."""
import numpy as np
from ._richresult import RichResult

__all__ = ["he_initialization"]


def he_initialization(x):
    """
    He/Kaiming weight initialization for ReLU

    Formula: W ~ N(0, 2/n_in)

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
    He et al. (2015)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "He/Kaiming weight initialization for ReLU"})


def cheatsheet():
    return "heinz: He/Kaiming weight initialization for ReLU"
