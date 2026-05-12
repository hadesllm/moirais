# morie.fn -- function file (hadesllm/morie)
"""Neurons-per-layer heuristic: typically similar width across hidden layers."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_neurons_per_layer"]


def geron_neurons_per_layer(n_features):
    """
    Neurons-per-layer heuristic: typically similar width across hidden layers

    Formula: width chosen in range [d, 2d] for d features

    Parameters
    ----------
    n_features : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: suggested_width

    References
    ----------
    Géron Ch 9
    """
    n_features = np.atleast_1d(np.asarray(n_features, dtype=float))
    n = len(n_features)
    result = float(np.mean(n_features))
    se = float(np.std(n_features, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Neurons-per-layer heuristic: typically similar width across hidden layers"})


def cheatsheet():
    return "hmnpl: Neurons-per-layer heuristic: typically similar width across hidden layers"
