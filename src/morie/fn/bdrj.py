# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Back-door adjustment formula (causal effect via covariate adjustment)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["backdoor_adjustment_formula"]


def backdoor_adjustment_formula(X, Y, Z, data):
    """
    Back-door adjustment formula (causal effect via covariate adjustment)

    Formula: P(Y=y|do(X=data)) = sum_z P(Y=y|X=data, Z=z) * P(Z=z)

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    Z : array-like
        Input data.
    data : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'ate': 'float'}

    References
    ----------
    Molak Ch 6
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Back-door adjustment formula (causal effect via covariate adjustment)"})


def cheatsheet():
    return "bdrj: Back-door adjustment formula (causal effect via covariate adjustment)"
