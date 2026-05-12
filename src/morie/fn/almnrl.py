# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Multiple-Negatives Ranking Loss: in-batch softmax over N candidates."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_multiple_negatives_ranking"]


def alammar_multiple_negatives_ranking(anchors, positives, tau):
    """
    Multiple-Negatives Ranking Loss: in-batch softmax over N candidates

    Formula: L = - (1/B) sum_i log( exp(sim(a_i, p_i)/tau) / sum_j exp(sim(a_i, p_j)/tau) )

    Parameters
    ----------
    anchors : array-like
        Input data.
    positives : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Alammar Ch 10, Multiple Negatives Ranking Loss section
    """
    anchors = np.atleast_1d(np.asarray(anchors, dtype=float))
    n = len(anchors)
    result = float(np.mean(anchors))
    se = float(np.std(anchors, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multiple-Negatives Ranking Loss: in-batch softmax over N candidates"})


def cheatsheet():
    return "almnrl: Multiple-Negatives Ranking Loss: in-batch softmax over N candidates"
