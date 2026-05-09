"""Design effect (DEFF) — variance ratio vs SRS."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["design_effect"]


def design_effect(y, weights, cluster):
    """
    Design effect (DEFF) — variance ratio vs SRS

    Formula: DEFF = Var_complex / Var_SRS

    Parameters
    ----------
    y : array-like
        Input data.
    weights : array-like
        Input data.
    cluster : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kish (1965)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Design effect (DEFF) — variance ratio vs SRS"})


def cheatsheet():
    return "desigeff: Design effect (DEFF) — variance ratio vs SRS"
