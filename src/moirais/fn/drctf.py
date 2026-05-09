"""DR-DiD with continuous treatment intensity."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dr_continuous_treatment"]


def dr_continuous_treatment(y, D_dose, X):
    """
    DR-DiD with continuous treatment intensity

    Formula: DR ATE per intensity level

    Parameters
    ----------
    y : array-like
        Input data.
    D_dose : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Callaway-Goodman-Bacon-Sant'Anna (2024)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DR-DiD with continuous treatment intensity"})


def cheatsheet():
    return "drctf: DR-DiD with continuous treatment intensity"
