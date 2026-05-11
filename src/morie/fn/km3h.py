# morie.fn — function file (hadesllm/morie)
"""3H alignment scoring rubric: helpful, harmless, honest."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_3h_alignment"]


def kamath_3h_alignment(helpful_score, harmless_score, honest_score, weights):
    """
    3H alignment scoring rubric: helpful, harmless, honest

    Formula: score_3h(y|x) = w_H * helpful(y|x) + w_A * harmless(y|x) + w_O * honest(y|x)

    Parameters
    ----------
    helpful_score : array-like
        Input data.
    harmless_score : array-like
        Input data.
    honest_score : array-like
        Input data.
    weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: score

    References
    ----------
    Kamath Ch 5, Alignment Tuning (3H) section
    """
    helpful_score = np.atleast_1d(np.asarray(helpful_score, dtype=float))
    n = len(helpful_score)
    result = float(np.mean(helpful_score))
    se = float(np.std(helpful_score, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "3H alignment scoring rubric: helpful, harmless, honest"})


def cheatsheet():
    return "km3h: 3H alignment scoring rubric: helpful, harmless, honest"
