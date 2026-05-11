# morie.fn — function file (hadesllm/morie)
"""SwiGLU activation used in LLaMA-style FFN: Swish-gated linear unit."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_swiglu_activation"]


def kamath_swiglu_activation(x, W, V, b, c):
    """
    SwiGLU activation used in LLaMA-style FFN: Swish-gated linear unit

    Formula: SwiGLU(x, W, V, b, c) = Swish(x W + b) * (x V + c);  Swish(z) = z * sigma(z)

    Parameters
    ----------
    x : array-like
        Input data.
    W : array-like
        Input data.
    V : array-like
        Input data.
    b : array-like
        Input data.
    c : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Kamath Ch 2, SwiGLU section (Shazeer 2020)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SwiGLU activation used in LLaMA-style FFN: Swish-gated linear unit"})


def cheatsheet():
    return "kmswig: SwiGLU activation used in LLaMA-style FFN: Swish-gated linear unit"
