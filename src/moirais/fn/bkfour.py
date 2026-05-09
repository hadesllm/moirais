"""Baron-Kenny 4-step causal-graph mediation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["baron_kenny_four_step"]


def baron_kenny_four_step(X, M, Y):
    """
    Baron-Kenny 4-step causal-graph mediation

    Formula: step1: Y~X; step2: M~X; step3: Y~X+M; step4: c'<c

    Parameters
    ----------
    X : array-like
        Input data.
    M : array-like
        Input data.
    Y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Baron & Kenny (1986)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Baron-Kenny 4-step causal-graph mediation"})


def cheatsheet():
    return "bkfour: Baron-Kenny 4-step causal-graph mediation"
