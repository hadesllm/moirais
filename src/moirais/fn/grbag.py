# moirais.fn — function file (hadesllm/moirais)
"""Bagging aggregator — mean of bootstrap-trained predictors."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_bagging_predictor"]


def geron_bagging_predictor(predictions):
    """
    Bagging aggregator — mean of bootstrap-trained predictors

    Formula: h_bag(x) = (1/B) sum_{b=1..B} h_b(x)

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
    Géron Ch 6, Bagging and Pasting section
    """
    predictions = np.asarray(predictions, dtype=float)
    n = int(predictions) if predictions.ndim == 0 else len(predictions)
    result = float(np.mean(predictions))
    se = float(np.std(predictions, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bagging aggregator — mean of bootstrap-trained predictors"})


def cheatsheet():
    return "grbag: Bagging aggregator — mean of bootstrap-trained predictors"
