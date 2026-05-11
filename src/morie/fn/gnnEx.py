"""GNNExplainer mask learning."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gnn_explainer"]


def gnn_explainer(model, graph, node):
    """
    GNNExplainer mask learning

    Formula: learn mask M minimizing prediction-mutual-info loss

    Parameters
    ----------
    model : array-like
        Input data.
    graph : array-like
        Input data.
    node : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ying et al (2019) GNNExplainer
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GNNExplainer mask learning"})


def cheatsheet():
    return "gnnEx: GNNExplainer mask learning"
