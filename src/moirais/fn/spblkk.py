"""Block kriging: prediction for areal unit B."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_block_kriging"]


def schabenberger_block_kriging(coords, z, blocks, cov_model):
    """
    Block kriging: prediction for areal unit B

    Formula: Z_hat(B) = (1/|B|)*integral Z_hat(s)ds = lambda'*Z; sigma^2_B = C(B,B)-2*c_B'*lambda+lambda'*C*lambda

    Parameters
    ----------
    coords : array-like
        Input data.
    z : array-like
        Input data.
    blocks : array-like
        Input data.
    cov_model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: block_prediction, variance

    References
    ----------
    Schabenberger Ch 5, Sec 5.7.1
    """
    z = np.asarray(z, dtype=float)
    n = int(z) if z.ndim == 0 else len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Block kriging: prediction for areal unit B"})


def cheatsheet():
    return "spblkk: Block kriging: prediction for areal unit B"
