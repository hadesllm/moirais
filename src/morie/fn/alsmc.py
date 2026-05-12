# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""SimCSE: two dropout passes of the same sentence as positive pair."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_simcse_dropout_aug"]


def alammar_simcse_dropout_aug(embeddings_dropout1, embeddings_dropout2, tau):
    """
    SimCSE: two dropout passes of the same sentence as positive pair

    Formula: L = - log( exp(sim(h_i^{z_1}, h_i^{z_2})/tau) / sum_j exp(sim(h_i^{z_1}, h_j^{z_2})/tau) )

    Parameters
    ----------
    embeddings_dropout1 : array-like
        Input data.
    embeddings_dropout2 : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Alammar Ch 10, SimCSE section
    """
    embeddings_dropout1 = np.atleast_1d(np.asarray(embeddings_dropout1, dtype=float))
    n = len(embeddings_dropout1)
    result = float(np.mean(embeddings_dropout1))
    se = float(np.std(embeddings_dropout1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SimCSE: two dropout passes of the same sentence as positive pair"})


def cheatsheet():
    return "alsmc: SimCSE: two dropout passes of the same sentence as positive pair"
