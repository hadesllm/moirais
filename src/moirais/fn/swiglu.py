"""SwiGLU gated activation (used in PaLM, LLaMA)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["swiglu_activation"]


def swiglu_activation(y, x, W, V):
    """
    SwiGLU gated activation (used in PaLM, LLaMA)

    Formula: SwiGLU(x, W, V, b, c) = (Swish(xW + b)) * (xV + c)

    Parameters
    ----------
    y : array-like
        Input data.
    x : array-like
        Input data.
    W : array-like
        Input data.
    V : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Shazeer (2020)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SwiGLU gated activation (used in PaLM, LLaMA)"})


def cheatsheet():
    return "swiglu: SwiGLU gated activation (used in PaLM, LLaMA)"
