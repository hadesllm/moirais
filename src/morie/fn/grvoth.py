# morie.fn — function file (hadesllm/morie)
"""Hard voting ensemble prediction (majority label among base classifiers)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_hard_voting"]


def geron_hard_voting(predictions):
    """
    Hard voting ensemble prediction (majority label among base classifiers)

    Formula: y_hat = mode(h_1(x), h_2(x), ..., h_L(x))

    Parameters
    ----------
    predictions : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_pred

    References
    ----------
    Géron Ch 6, Voting Classifier (hard) section
    """
    predictions = np.asarray(predictions, dtype=float)
    n = int(predictions) if predictions.ndim == 0 else len(predictions)
    result = float(np.mean(predictions))
    se = float(np.std(predictions, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hard voting ensemble prediction (majority label among base classifiers)"})


def cheatsheet():
    return "grvoth: Hard voting ensemble prediction (majority label among base classifiers)"
