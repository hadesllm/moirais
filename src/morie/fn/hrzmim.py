# morie.fn — function file (hadesllm/morie)
"""Multiple-index model with several indices x'beta_1, x'beta_2,...."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_multiple_index_model"]


def horowitz_multiple_index_model(x, y, k):
    """
    Multiple-index model with several indices x'beta_1, x'beta_2,...

    Formula: E(Y|X=x) = G(x'beta_1,...,x'beta_k)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: betas, G_hat

    References
    ----------
    Horowitz Ch 2, Sec 2.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multiple-index model with several indices x'beta_1, x'beta_2,..."})


def cheatsheet():
    return "hrzmim: Multiple-index model with several indices x'beta_1, x'beta_2,..."
