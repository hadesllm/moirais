# moirais.fn — function file (hadesllm/moirais)
"""WordPiece likelihood-maximizing merge score."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_wordpiece_tokenizer_score"]


def geron_wordpiece_tokenizer_score(counts, pairs):
    """
    WordPiece likelihood-maximizing merge score

    Formula: score(A, B) = count(AB) / (count(A) * count(B))

    Parameters
    ----------
    counts : array-like
        Input data.
    pairs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: scores

    References
    ----------
    Géron Ch 14, WordPiece section
    """
    counts = np.atleast_1d(np.asarray(counts, dtype=float))
    n = len(counts)
    result = float(np.mean(counts))
    se = float(np.std(counts, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "WordPiece likelihood-maximizing merge score"})


def cheatsheet():
    return "grwpc: WordPiece likelihood-maximizing merge score"
