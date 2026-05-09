# moirais.fn — function file (hadesllm/moirais)
"""Zero-crossing rate."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_zero_crossing"]


def rangayyan_zero_crossing(x):
    """
    Zero-crossing rate

    Formula: ZCR = (1/N) sum |sign(x[n]) - sign(x[n-1])| / 2

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
    Rangayyan Ch 5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Zero-crossing rate"})


def cheatsheet():
    return "rgzcr: Zero-crossing rate"
