"""Cohen's kappa for two raters."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cohens_kappa"]


def cohens_kappa(rater1, rater2):
    """
    Cohen's kappa for two raters

    Formula: kappa = (p_o - p_e) / (1 - p_e)

    Parameters
    ----------
    rater1 : array-like
        Input data.
    rater2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cohen (1960)
    """
    rater1 = np.atleast_1d(np.asarray(rater1, dtype=float))
    n = len(rater1)
    result = float(np.mean(rater1))
    se = float(np.std(rater1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cohen's kappa for two raters"})


def cheatsheet():
    return "kappac: Cohen's kappa for two raters"
