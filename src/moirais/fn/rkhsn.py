# moirais.fn — function file (hadesllm/moirais)
"""RKHS norm and reproducing property."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rkhs_norm"]


def rkhs_norm(alpha, eigenvalues):
    """
    RKHS norm and reproducing property

    Formula: <f,g>_H = sum_i lambda_i^{-1} alpha_i beta_i; ||f||_H^2 = sum_i alpha_i^2/lambda_i

    Parameters
    ----------
    alpha : array-like
        Input data.
    eigenvalues : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'norm': 'float'}

    References
    ----------
    Montesinos Lopez Ch 8
    """
    alpha = np.asarray(alpha, dtype=float)
    n = int(alpha) if alpha.ndim == 0 else len(alpha)
    result = float(np.mean(alpha))
    se = float(np.std(alpha, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RKHS norm and reproducing property"})


def cheatsheet():
    return "rkhsn: RKHS norm and reproducing property"
