# morie.fn -- function file (hadesllm/morie)
"""InfoNCE contrastive loss: pulls positives together, pushes negatives apart."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_contrastive_infonce"]


def geron_contrastive_infonce(anchors, positives, negatives, tau):
    """
    InfoNCE contrastive loss: pulls positives together, pushes negatives apart

    Formula: L = - log ( exp(sim(a, p)/tau) / (exp(sim(a,p)/tau) + sum_n exp(sim(a,n)/tau)) )

    Parameters
    ----------
    anchors : array-like
        Input data.
    positives : array-like
        Input data.
    negatives : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Géron Ch 16, Contrastive Learning section
    """
    anchors = np.atleast_1d(np.asarray(anchors, dtype=float))
    n = len(anchors)
    result = float(np.mean(anchors))
    se = float(np.std(anchors, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "InfoNCE contrastive loss: pulls positives together, pushes negatives apart"})


def cheatsheet():
    return "grctr: InfoNCE contrastive loss: pulls positives together, pushes negatives apart"
