"""Complete spatial randomness (CSR): homogeneous Poisson process."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_csr_def"]


def schabenberger_csr_def(points, region):
    """
    Complete spatial randomness (CSR): homogeneous Poisson process

    Formula: N(A) ~ Poisson(lambda*|A|), points independent of each other

    Parameters
    ----------
    points : array-like
        Input data.
    region : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: boolean

    References
    ----------
    Schabenberger Ch 3
    """
    points = np.asarray(points, dtype=float)
    n = int(points) if points.ndim == 0 else len(points)
    result = float(np.mean(points))
    se = float(np.std(points, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Complete spatial randomness (CSR): homogeneous Poisson process"})


def cheatsheet():
    return "spcsr: Complete spatial randomness (CSR): homogeneous Poisson process"
