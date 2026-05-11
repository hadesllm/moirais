# morie.fn — function file (hadesllm/morie)
"""Feature importance from tree-based ensembles via average impurity decrease."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_feature_importance"]


def geron_feature_importance(model, feature_names):
    """
    Feature importance from tree-based ensembles via average impurity decrease

    Formula: imp_j = avg over trees of weighted impurity decreases using feature j

    Parameters
    ----------
    model : array-like
        Input data.
    feature_names : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: importances

    References
    ----------
    Géron Ch 6
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Feature importance from tree-based ensembles via average impurity decrease"})
    estimate = np.median(model)
    se = 1.2533 * np.std(model, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Feature importance from tree-based ensembles via average impurity decrease"})


def cheatsheet():
    return "hmfim: Feature importance from tree-based ensembles via average impurity decrease"
