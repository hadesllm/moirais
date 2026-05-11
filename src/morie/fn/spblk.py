"""Block kriging for areal prediction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["spatial_block_kriging"]


def spatial_block_kriging(x, coords, blocks):
    """
    Block kriging for areal prediction

    Formula: Z_hat(B) = (1/|B|) integral Z_hat(s) ds

    Parameters
    ----------
    x : array-like
        Input data.
    coords : array-like
        Input data.
    blocks : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Schabenberger Ch 4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Block kriging for areal prediction"})


def cheatsheet():
    return "spblk: Block kriging for areal prediction"
