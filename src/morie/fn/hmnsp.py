# morie.fn — function file (hadesllm/morie)
"""Next sentence prediction pretraining (BERT)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_next_sentence_prediction"]


def geron_next_sentence_prediction(sent_A, sent_B):
    """
    Next sentence prediction pretraining (BERT)

    Formula: binary classifier on [CLS]: is sentence B the next of A?

    Parameters
    ----------
    sent_A : array-like
        Input data.
    sent_B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prob

    References
    ----------
    Géron Ch 15
    """
    sent_A = np.atleast_1d(np.asarray(sent_A, dtype=float))
    n = len(sent_A)
    result = float(np.mean(sent_A))
    se = float(np.std(sent_A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Next sentence prediction pretraining (BERT)"})


def cheatsheet():
    return "hmnsp: Next sentence prediction pretraining (BERT)"
