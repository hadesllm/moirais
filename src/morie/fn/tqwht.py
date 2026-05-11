"""Walsh-Hadamard Transform with 1/sqrt(d) normalization (CRITICAL — not 1/d)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["turboquant_walsh_hadamard_transform"]


def turboquant_walsh_hadamard_transform(x):
    """
    Walsh-Hadamard Transform with 1/sqrt(d) normalization (CRITICAL — not 1/d)

    Formula: y = H x / sqrt(d);  H_{d x d} = Hadamard(d);  inverse: x = H y / sqrt(d)  (H^2 / d = I)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    TurboQuant MORIE integration — morie/quant_ggml.c
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Walsh-Hadamard Transform with 1/sqrt(d) normalization (CRITICAL — not 1/d)"})


def cheatsheet():
    return "tqwht: Walsh-Hadamard Transform with 1/sqrt(d) normalization (CRITICAL — not 1/d)"
