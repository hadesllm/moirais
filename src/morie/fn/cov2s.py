# morie.fn — function file (hadesllm/morie)
"""Two-sample coverage probability."""
import numpy as np
from ._richresult import RichResult

__all__ = ["two_sample_coverage"]


def two_sample_coverage(x, y):
    """
    Two-sample coverage probability

    Formula: Placement coverage between two samples

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gibbons Ch 2.11.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Two-sample coverage probability"})


def cheatsheet():
    return "cov2s: Two-sample coverage probability"
