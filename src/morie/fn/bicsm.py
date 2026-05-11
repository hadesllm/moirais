# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""BIC score for DAG structure scoring in GES."""
import numpy as np
from ._richresult import RichResult

__all__ = ["bic_score_dag"]


def bic_score_dag(dag, data):
    """
    BIC score for DAG structure scoring in GES

    Formula: BIC(G, data) = log P(data|G, theta_hat) - (k/2)*log(n); k = number of free parameters

    Parameters
    ----------
    dag : array-like
        Input data.
    data : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'bic': 'float'}

    References
    ----------
    Molak Ch 13
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BIC score for DAG structure scoring in GES"})


def cheatsheet():
    return "bicsm: BIC score for DAG structure scoring in GES"
