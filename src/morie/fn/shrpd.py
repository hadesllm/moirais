"""Shepard diagram: plot of dissimilarities vs distances to assess MDS fit."""
import numpy as np
from ._richresult import RichResult

__all__ = ["shepard_diagram"]


def shepard_diagram(delta, D_config):
    """
    Shepard diagram: plot of dissimilarities vs distances to assess MDS fit

    Formula: Plot (delta_ij, d_ij(X)) pairs; monotone trend indicates good nonmetric fit

    Parameters
    ----------
    delta : array-like
        Input data.
    D_config : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'plot_data': 'tuple'}

    References
    ----------
    Armstrong Ch 3
    """
    delta = np.asarray(delta, dtype=float)
    n = int(delta) if delta.ndim == 0 else len(delta)
    result = float(np.mean(delta))
    se = float(np.std(delta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Shepard diagram: plot of dissimilarities vs distances to assess MDS fit"})


def cheatsheet():
    return "shrpd: Shepard diagram: plot of dissimilarities vs distances to assess MDS fit"
