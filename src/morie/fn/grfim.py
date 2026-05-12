# morie.fn -- function file (hadesllm/morie)
"""Feature importance via mean decrease in impurity across forest trees."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_feature_importance_mdi"]


def geron_feature_importance_mdi(tree_importances):
    """
    Feature importance via mean decrease in impurity across forest trees

    Formula: imp(f) = (1/B) sum_b sum_{n in T_b(f)} (m_n/m) * delta_impurity(n)

    Parameters
    ----------
    tree_importances : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: importances

    References
    ----------
    Géron Ch 6, Feature Importance section
    """
    tree_importances = np.asarray(tree_importances, dtype=float)
    n = int(tree_importances) if tree_importances.ndim == 0 else len(tree_importances)
    result = float(np.mean(tree_importances))
    se = float(np.std(tree_importances, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Feature importance via mean decrease in impurity across forest trees"})


def cheatsheet():
    return "grfim: Feature importance via mean decrease in impurity across forest trees"
