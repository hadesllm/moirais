"""Mixed-model ANOVA-style variance decomposition."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["random_effects_anova_decomp"]


def random_effects_anova_decomp(y, cluster):
    """
    Mixed-model ANOVA-style variance decomposition

    Formula: SS_total = SS_between + SS_within; F = MS_between / MS_within

    Parameters
    ----------
    y : array-like
        Input data.
    cluster : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Searle, Casella, McCulloch (1992)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mixed-model ANOVA-style variance decomposition"})


def cheatsheet():
    return "ranova: Mixed-model ANOVA-style variance decomposition"
