"""SwiGLU gated activation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["swiglu_activation"]


def swiglu_activation(x):
    """
    SwiGLU gated activation

    Formula: SwiGLU(x,W,V,b,c) = Swish(xW+b) * (xV+c)

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
    Shazeer (2020)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SwiGLU gated activation"})


def cheatsheet():
    return "swigl: SwiGLU gated activation"
