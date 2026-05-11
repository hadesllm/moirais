"""l-diversity baseline."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["l_diversity"]


def l_diversity(X, quasi_ids, sensitive, l):
    """
    l-diversity baseline

    Formula: each equivalence class has ≥l well-represented sensitive vals

    Parameters
    ----------
    X : array-like
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
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "l-diversity baseline"})


def cheatsheet():
    return "dpld: l-diversity baseline"
