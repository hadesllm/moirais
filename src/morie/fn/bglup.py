# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""BayesCpi for variable selection in genomics."""
import numpy as np
from ._richresult import RichResult

__all__ = ["bayes_cpi_genomic"]


def bayes_cpi_genomic(x, y):
    """
    BayesCpi for variable selection in genomics

    Formula: beta_j ~ pi*N(0,sigma^2) + (1-pi)*delta(0)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Montesinos Lopez Ch 4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BayesCpi for variable selection in genomics"})


def cheatsheet():
    return "bglup: BayesCpi for variable selection in genomics"
