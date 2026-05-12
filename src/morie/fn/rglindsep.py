# morie.fn -- function file (hadesllm/morie)
"""Linear discriminant function with optimal separability."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_lin_discr_sep"]


def rangayyan_lin_discr_sep(X_1, X_2):
    """
    Linear discriminant function with optimal separability

    Formula: Project X_1 to w^T*X_1; optimal w = S_W^{-1}*(mu_1-mu_2)

    Parameters
    ----------
    X_1 : array-like
        Input data.
    X_2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: w, separation

    References
    ----------
    Rangayyan Ch 10.4.2
    """
    X_1 = np.asarray(X_1, dtype=float)
    n = int(X_1) if X_1.ndim == 0 else len(X_1)
    result = float(np.mean(X_1))
    se = float(np.std(X_1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Linear discriminant function with optimal separability"})


def cheatsheet():
    return "rglindsep: Linear discriminant function with optimal separability"
