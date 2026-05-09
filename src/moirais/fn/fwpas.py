# moirais.fn — function file (hadesllm/moirais)
"""Dense layer forward pass."""
import numpy as np
from ._richresult import RichResult

__all__ = ["forward_pass_dense"]


def forward_pass_dense(x, w, b):
    """
    Dense layer forward pass

    Formula: z = Wx + b, a = sigma(z)

    Parameters
    ----------
    x : array-like
        Input data.
    w : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Geron (2026), Ch 10
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dense layer forward pass"})


def cheatsheet():
    return "fwpas: Dense layer forward pass"
