# morie.fn — function file (hadesllm/morie)
"""Pearl's Ladder of Causation: three rungs (association, intervention, counterfactual)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ladder_of_causation"]


def ladder_of_causation(rung, model):
    """
    Pearl's Ladder of Causation: three rungs (association, intervention, counterfactual)

    Formula: Rung 1: P(Y|X); Rung 2: P(Y|do(X=rung)); Rung 3: P(Y_x | X=rung', Y=y')

    Parameters
    ----------
    rung : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'query': 'distribution'}

    References
    ----------
    Molak Ch 2
    """
    rung = np.asarray(rung, dtype=float)
    y = np.asarray(model, dtype=float)
    n = min(len(rung), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Pearl's Ladder of Causation: three rungs (association, intervention, counterfactual)"})
    result = stats.spearmanr(rung[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Pearl's Ladder of Causation: three rungs (association, intervention, counterfactual)"})


def cheatsheet():
    return "ladrc: Pearl's Ladder of Causation: three rungs (association, intervention, counterfactual)"
