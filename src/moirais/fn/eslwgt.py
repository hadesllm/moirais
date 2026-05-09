"""Weight decay (L2) penalty."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_weight_decay"]


def esl_weight_decay(weights, lambda_):
    """
    Weight decay (L2) penalty

    Formula: Loss + lambda sum w^2

    Parameters
    ----------
    weights : array-like
        Input data.
    lambda_ : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Hastie ESL Ch 11
    """
    weights = np.atleast_1d(np.asarray(weights, dtype=float))
    n = len(weights)
    result = float(np.mean(weights))
    se = float(np.std(weights, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Weight decay (L2) penalty"})


def cheatsheet():
    return "eslwgt: Weight decay (L2) penalty"
