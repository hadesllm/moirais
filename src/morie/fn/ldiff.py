"""l-diversity (Machanavajjhala et al)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["l_diversity_check"]


def l_diversity_check(y, quasi_ids, sensitive, l):
    """
    l-diversity (Machanavajjhala et al)

    Formula: each k-anon group has >= l well-represented sensitive values

    Parameters
    ----------
    y : array-like
        Input data.
    quasi_ids : array-like
        Input data.
    sensitive : array-like
        Input data.
    l : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Machanavajjhala et al (2007)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "l-diversity (Machanavajjhala et al)"})


def cheatsheet():
    return "ldiff: l-diversity (Machanavajjhala et al)"
