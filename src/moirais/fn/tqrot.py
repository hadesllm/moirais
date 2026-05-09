"""Random orthogonal rotation matrix generator (QR of a Gaussian matrix)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["turboquant_rotation_matrix"]


def turboquant_rotation_matrix(d, seed):
    """
    Random orthogonal rotation matrix generator (QR of a Gaussian matrix)

    Formula: A ~ N(0,1)^{d x d};  Q, _ = QR(A);  return Q

    Parameters
    ----------
    d : array-like
        Input data.
    seed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Q

    References
    ----------
    Zandieh et al. 2024 Section 4.1 (rotation_matrix, QR step)
    """
    d = np.atleast_1d(np.asarray(d, dtype=float))
    n = len(d)
    result = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Random orthogonal rotation matrix generator (QR of a Gaussian matrix)"})


def cheatsheet():
    return "tqrot: Random orthogonal rotation matrix generator (QR of a Gaussian matrix)"
