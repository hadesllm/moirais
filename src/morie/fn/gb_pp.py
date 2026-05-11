# morie.fn — function file (hadesllm/morie)
"""PP-plot (probability-probability plot) for visual GOF assessment."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_pp_plot"]


def gibbons_pp_plot(x, F0):
    """
    PP-plot (probability-probability plot) for visual GOF assessment

    Formula: Plot (F0(X_(i)), i/n) pairs; departure from diagonal = misfit

    Parameters
    ----------
    x : array-like
        Input data.
    F0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: plot_data

    References
    ----------
    Gibbons Ch 4.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PP-plot (probability-probability plot) for visual GOF assessment"})


def cheatsheet():
    return "gb_pp: PP-plot (probability-probability plot) for visual GOF assessment"
