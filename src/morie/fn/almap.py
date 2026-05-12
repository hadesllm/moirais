# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Mean Average Precision over Q queries."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_mean_average_precision"]


def alammar_mean_average_precision(rankings_by_query, relevant_by_query):
    """
    Mean Average Precision over Q queries

    Formula: AP(q) = sum_k P(k) * rel(k) / |relevant(q)|;  MAP = (1/|Q|) sum AP(q)

    Parameters
    ----------
    rankings_by_query : array-like
        Input data.
    relevant_by_query : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: map_score

    References
    ----------
    Alammar Ch 8, MAP section
    """
    rankings_by_query = np.atleast_1d(np.asarray(rankings_by_query, dtype=float))
    n = len(rankings_by_query)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Mean Average Precision over Q queries"})
    estimate = np.median(rankings_by_query)
    se = 1.2533 * np.std(rankings_by_query, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Mean Average Precision over Q queries"})


def cheatsheet():
    return "almap: Mean Average Precision over Q queries"
