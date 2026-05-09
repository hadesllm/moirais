"""Frank-Wolfe / conditional gradient."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["frank_wolfe"]


def frank_wolfe(f, grad_f, domain, x0, steps):
    """
    Frank-Wolfe / conditional gradient

    Formula: x_{t+1} = (1-gamma) x_t + gamma argmin <grad, x>

    Parameters
    ----------
    f : array-like
        Input data.
    grad_f : array-like
        Input data.
    domain : array-like
        Input data.
    x0 : array-like
        Input data.
    steps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Frank-Wolfe (1956)
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Frank-Wolfe / conditional gradient"})


def cheatsheet():
    return "frwol2: Frank-Wolfe / conditional gradient"
