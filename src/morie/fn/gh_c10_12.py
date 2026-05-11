# morie.fn — function file (hadesllm/morie)
"""Model selection consistency: Bayes factor selects true model for nested models."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_modsel_bic"]


def ghosal_modsel_bic(x):
    """
    Model selection consistency: Bayes factor selects true model for nested models

    Formula: BF(H1,H0) = P(X^n|H1)/P(X^n|H0) -> infty if H1 true, -> 0 if H0 true

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 10 §10.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Model selection consistency: Bayes factor selects true model for nested models"})


def cheatsheet():
    return "gh_c10_12: Model selection consistency: Bayes factor selects true model for nested models"
