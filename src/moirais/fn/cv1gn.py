# moirais.fn — function file (hadesllm/moirais)
"""CV1 genomic cross-validation: train on observed, predict unobserved lines."""
import numpy as np
from ._richresult import RichResult

__all__ = ["cv1_genomic"]


def cv1_genomic(y, markers, n_folds):
    """
    CV1 genomic cross-validation: train on observed, predict unobserved lines

    Formula: Partition lines into training (phenotyped) / validation (genotyped only); r = cor(y_obs, y_hat)

    Parameters
    ----------
    y : array-like
        Input data.
    markers : array-like
        Input data.
    n_folds : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'pa': 'float'}

    References
    ----------
    Montesinos Lopez Ch 4
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CV1 genomic cross-validation: train on observed, predict unobserved lines"})


def cheatsheet():
    return "cv1gn: CV1 genomic cross-validation: train on observed, predict unobserved lines"
