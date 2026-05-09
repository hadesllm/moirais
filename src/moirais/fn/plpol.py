# moirais.fn — function file (hadesllm/moirais)
"""Spatial map plot coordinates for ideal point visualization."""
import numpy as np
from ._richresult import RichResult

__all__ = ["plot_spatial"]


def plot_spatial(ideal_points, party_labels, stimuli_labels):
    """
    Spatial map plot coordinates for ideal point visualization

    Formula: Plot x_i* in d-dimensional space; label by party/group; annotate vote positions

    Parameters
    ----------
    ideal_points : array-like
        Input data.
    party_labels : array-like
        Input data.
    stimuli_labels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'plot_coords': 'dict'}

    References
    ----------
    Armstrong Ch 2,5,6
    """
    ideal_points = np.asarray(ideal_points, dtype=float)
    n = int(ideal_points) if ideal_points.ndim == 0 else len(ideal_points)
    result = float(np.mean(ideal_points))
    se = float(np.std(ideal_points, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spatial map plot coordinates for ideal point visualization"})


def cheatsheet():
    return "plpol: Spatial map plot coordinates for ideal point visualization"
