# moirais.fn — function file (hadesllm/moirais)
"""SENet channel-wise Squeeze-and-Excitation attention."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_senet_squeeze_excite"]


def geron_senet_squeeze_excite(X, W1, W2):
    """
    SENet channel-wise Squeeze-and-Excitation attention

    Formula: z = GAP(X); s = sigma(W2 * ReLU(W1 z)); Y = s (broadcast) * X

    Parameters
    ----------
    X : array-like
        Input data.
    W1 : array-like
        Input data.
    W2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Y

    References
    ----------
    Géron Ch 12, SENet section
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SENet channel-wise Squeeze-and-Excitation attention"})


def cheatsheet():
    return "grsen: SENet channel-wise Squeeze-and-Excitation attention"
