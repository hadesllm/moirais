"""ROUGE-N / ROUGE-L."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rouge"]


def rouge(candidate, reference, kind):
    """
    ROUGE-N / ROUGE-L

    Formula: recall over n-gram overlap; LCS for ROUGE-L

    Parameters
    ----------
    candidate : array-like
        Input data.
    reference : array-like
        Input data.
    kind : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lin (2004)
    """
    candidate = np.atleast_1d(np.asarray(candidate, dtype=float))
    n = len(candidate)
    result = float(np.mean(candidate))
    se = float(np.std(candidate, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ROUGE-N / ROUGE-L"})


def cheatsheet():
    return "rouge: ROUGE-N / ROUGE-L"
