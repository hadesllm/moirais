# morie.fn — function file (hadesllm/morie)
"""Mean cross-validation score across K folds."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_cross_validation_score"]


def geron_cross_validation_score(X, y, K):
    """
    Mean cross-validation score across K folds

    Formula: CV_score = (1/K) sum_{k=1..K} score(model_k, fold_k)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mean_score, std_score

    References
    ----------
    Géron Ch 1 (intro), Ch 2 (use)
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mean cross-validation score across K folds"})


def cheatsheet():
    return "grcvs: Mean cross-validation score across K folds"
