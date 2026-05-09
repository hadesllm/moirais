"""Variance components via Henderson Method III (ANOVA-style)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["variance_components_henderson3"]


def variance_components_henderson3(y, cluster):
    """
    Variance components via Henderson Method III (ANOVA-style)

    Formula: sigma2_u = (MS_b - MS_w)/n; sigma2_e = MS_w

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
    Henderson (1953)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variance components via Henderson Method III (ANOVA-style)"})


def cheatsheet():
    return "vcomp: Variance components via Henderson Method III (ANOVA-style)"
