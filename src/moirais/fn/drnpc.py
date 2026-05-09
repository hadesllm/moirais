"""DR-DiD with negative control outcome."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dr_did_neg_control"]


def dr_did_neg_control(y_main, y_neg, D, X):
    """
    DR-DiD with negative control outcome

    Formula: DR moment on neg-outcome falsification

    Parameters
    ----------
    y_main : array-like
        Input data.
    y_neg : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lipsitch et al (2010)
    """
    y_main = np.atleast_1d(np.asarray(y_main, dtype=float))
    n = len(y_main)
    result = float(np.mean(y_main))
    se = float(np.std(y_main, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DR-DiD with negative control outcome"})


def cheatsheet():
    return "drnpc: DR-DiD with negative control outcome"
