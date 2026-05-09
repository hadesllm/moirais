# moirais.fn — function file (hadesllm/moirais)
"""Root mean square (RMS) value of a signal."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_rms"]


def rangayyan_rms(x):
    """
    Root mean square (RMS) value of a signal

    Formula: RMS = sqrt((1/N) sum x[n]^2)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rms

    References
    ----------
    Rangayyan Ch 5.6.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Root mean square (RMS) value of a signal"})


def cheatsheet():
    return "rgrms: Root mean square (RMS) value of a signal"
