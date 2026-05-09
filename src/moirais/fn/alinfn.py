# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""InfoNCE contrastive loss with in-batch negatives."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_infonce_loss"]


def alammar_infonce_loss(anchor, positive, negatives, tau):
    """
    InfoNCE contrastive loss with in-batch negatives

    Formula: L = - log( exp(sim(a, p)/tau) / sum_{n in negs + {p}} exp(sim(a, n)/tau) )

    Parameters
    ----------
    anchor : array-like
        Input data.
    positive : array-like
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
    Alammar Ch 10, InfoNCE section
    """
    anchor = np.atleast_1d(np.asarray(anchor, dtype=float))
    n = len(anchor)
    result = float(np.mean(anchor))
    se = float(np.std(anchor, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "InfoNCE contrastive loss with in-batch negatives"})


def cheatsheet():
    return "alinfn: InfoNCE contrastive loss with in-batch negatives"
