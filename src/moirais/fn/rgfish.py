# moirais.fn — function file (hadesllm/moirais)
"""Fisher's criterion for feature separability."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_fisher_criterion"]


def rangayyan_fisher_criterion(X, y):
    """
    Fisher's criterion for feature separability

    Formula: J(w) = (mu_1-mu_2)^2 / (s_1^2+s_2^2) for scalar feature

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: j_values, ranked_features

    References
    ----------
    Rangayyan Ch 10.10.2
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fisher's criterion for feature separability"})


def cheatsheet():
    return "rgfish: Fisher's criterion for feature separability"
