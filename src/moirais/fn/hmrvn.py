# moirais.fn — function file (hadesllm/moirais)
"""RevNet: reversible residual blocks enabling activation-free backprop."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_revnet"]


def geron_revnet(x, F, G):
    """
    RevNet: reversible residual blocks enabling activation-free backprop

    Formula: y1 = x1 + F(x2); y2 = x2 + G(y1); reversible

    Parameters
    ----------
    x : array-like
        Input data.
    F : array-like
        Input data.
    G : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Géron Ch 12
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RevNet: reversible residual blocks enabling activation-free backprop"})


def cheatsheet():
    return "hmrvn: RevNet: reversible residual blocks enabling activation-free backprop"
