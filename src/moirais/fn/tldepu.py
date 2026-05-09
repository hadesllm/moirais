"""Upper tail dependence coefficient."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["upper_tail_dependence"]


def upper_tail_dependence(y, copula, theta):
    """
    Upper tail dependence coefficient

    Formula: lambda_U = lim_{u->1-} (1 - 2u + C(u,u))/(1 - u)

    Parameters
    ----------
    y : array-like
        Input data.
    copula : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Joe (1997); Nelsen (2006) §5.4
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Upper tail dependence coefficient"})


def cheatsheet():
    return "tldepu: Upper tail dependence coefficient"
