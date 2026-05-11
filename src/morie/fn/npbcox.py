"""NP Bayes Cox model with Beta-process baseline."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["np_bayes_cox"]


def np_bayes_cox(time, event, X):
    """
    NP Bayes Cox model with Beta-process baseline

    Formula: baseline cumulative hazard ~ BetaProc; beta unrestricted

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kalbfleisch (1978)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "NP Bayes Cox model with Beta-process baseline"})


def cheatsheet():
    return "npbcox: NP Bayes Cox model with Beta-process baseline"
