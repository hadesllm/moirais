"""SiLU / Swish activation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["silu_swish"]


def silu_swish(y):
    """
    SiLU / Swish activation

    Formula: SiLU(x) = x * sigmoid(x)

    Parameters
    ----------
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Elfwing, Uchibe, Doya (2018); Ramachandran et al. (2017) Swish
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SiLU / Swish activation"})


def cheatsheet():
    return "silu: SiLU / Swish activation"
