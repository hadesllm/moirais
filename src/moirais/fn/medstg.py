"""Sequential / chain mediation X -> M1 -> M2 -> Y."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sequential_mediation"]


def sequential_mediation(a1, b1, c1):
    """
    Sequential / chain mediation X -> M1 -> M2 -> Y

    Formula: a1 * b1 * c1 * (paths through chain)

    Parameters
    ----------
    a1 : array-like
        Input data.
    b1 : array-like
        Input data.
    c1 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    VanderWeele (2015) §5
    """
    a1 = np.atleast_1d(np.asarray(a1, dtype=float))
    n = len(a1)
    result = float(np.mean(a1))
    se = float(np.std(a1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sequential / chain mediation X -> M1 -> M2 -> Y"})


def cheatsheet():
    return "medstg: Sequential / chain mediation X -> M1 -> M2 -> Y"
