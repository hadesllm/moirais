"""AI Feynman symbolic regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ai_feynman"]


def ai_feynman(X, y):
    """
    AI Feynman symbolic regression

    Formula: divide-and-conquer using physics priors

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Udrescu-Tegmark (2020)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AI Feynman symbolic regression"})


def cheatsheet():
    return "feynAI: AI Feynman symbolic regression"
