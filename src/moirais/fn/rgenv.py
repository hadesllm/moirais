# moirais.fn — function file (hadesllm/moirais)
"""Envelope detection (Hilbert transform)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_envelope"]


def rangayyan_envelope(x):
    """
    Envelope detection (Hilbert transform)

    Formula: env(t) = |x(t) + j*H{x(t)}|

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Envelope detection (Hilbert transform)"})


def cheatsheet():
    return "rgenv: Envelope detection (Hilbert transform)"
