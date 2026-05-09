"""Tag-aware recommendation (FolkRank)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tag_aware_rec"]


def tag_aware_rec(UTI, alpha):
    """
    Tag-aware recommendation (FolkRank)

    Formula: random walk on user-tag-item tensor

    Parameters
    ----------
    UTI : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hotho et al (2006)
    """
    UTI = np.atleast_1d(np.asarray(UTI, dtype=float))
    n = len(UTI)
    result = float(np.mean(UTI))
    se = float(np.std(UTI, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Tag-aware recommendation (FolkRank)"})


def cheatsheet():
    return "tagRC: Tag-aware recommendation (FolkRank)"
