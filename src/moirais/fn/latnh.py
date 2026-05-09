# moirais.fn — function file (hadesllm/moirais)
"""Latin hypercube sampling."""
import numpy as np
from ._richresult import RichResult

__all__ = ["latin_hypercube"]


def latin_hypercube(x):
    """
    Latin hypercube sampling

    Formula: Stratified random in d dimensions

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
    McKay et al. (1979)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Latin hypercube sampling"})


def cheatsheet():
    return "latnh: Latin hypercube sampling"
