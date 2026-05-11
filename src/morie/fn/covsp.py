# morie.fn — function file (hadesllm/morie)
"""One-sample coverage probability."""
import numpy as np
from ._richresult import RichResult

__all__ = ["one_sample_coverage"]


def one_sample_coverage(x):
    """
    One-sample coverage probability

    Formula: C = F(X_(s)) - F(X_(r))

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
    Gibbons Ch 2.11.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "One-sample coverage probability"})


def cheatsheet():
    return "covsp: One-sample coverage probability"
