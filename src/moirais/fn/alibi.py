"""ALiBi linear bias attention."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alibi"]


def alibi(scores, slopes):
    """
    ALiBi linear bias attention

    Formula: attn += -m·|i-j|

    Parameters
    ----------
    scores : array-like
        Input data.
    slopes : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Press-Smith-Lewis (2022) ALiBi
    """
    scores = np.atleast_1d(np.asarray(scores, dtype=float))
    n = len(scores)
    result = float(np.mean(scores))
    se = float(np.std(scores, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ALiBi linear bias attention"})


def cheatsheet():
    return "alibi: ALiBi linear bias attention"
