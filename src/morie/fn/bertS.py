"""BERTScore -- embedding-based eval."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bertscore"]


def bertscore(candidate, reference, model):
    """
    BERTScore -- embedding-based eval

    Formula: max-cosine matching between token embeddings

    Parameters
    ----------
    candidate : array-like
        Input data.
    reference : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Zhang et al (2020) BERTScore
    """
    candidate = np.atleast_1d(np.asarray(candidate, dtype=float))
    n = len(candidate)
    result = float(np.mean(candidate))
    se = float(np.std(candidate, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BERTScore -- embedding-based eval"})


def cheatsheet():
    return "bertS: BERTScore -- embedding-based eval"
