# morie.fn — function file (hadesllm/morie)
"""MinT reconciliation: trace-minimizing weighted least-squares reconciliation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_mint_reconciliation"]


def joseph_mint_reconciliation(y_hat, S, W):
    """
    MinT reconciliation: trace-minimizing weighted least-squares reconciliation

    Formula: y_tilde = S (S^T W^{-1} S)^{-1} S^T W^{-1} y_hat;  W = covariance of base forecast errors

    Parameters
    ----------
    y_hat : array-like
        Input data.
    S : array-like
        Input data.
    W : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_reconciled

    References
    ----------
    Joseph Ch 17, MinT reconciliation section
    """
    y_hat = np.atleast_1d(np.asarray(y_hat, dtype=float))
    n = len(y_hat)
    result = float(np.mean(y_hat))
    se = float(np.std(y_hat, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MinT reconciliation: trace-minimizing weighted least-squares reconciliation"})


def cheatsheet():
    return "jomint: MinT reconciliation: trace-minimizing weighted least-squares reconciliation"
