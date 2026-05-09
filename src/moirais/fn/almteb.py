# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""MTEB aggregate: mean score across task categories."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_mteb_benchmark_score"]


def alammar_mteb_benchmark_score(task_scores, category_map):
    """
    MTEB aggregate: mean score across task categories

    Formula: MTEB = mean over categories: mean over tasks in category

    Parameters
    ----------
    task_scores : array-like
        Input data.
    category_map : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: aggregate

    References
    ----------
    Alammar Ch 10, MTEB section
    """
    task_scores = np.atleast_1d(np.asarray(task_scores, dtype=float))
    n = len(task_scores)
    result = float(np.mean(task_scores))
    se = float(np.std(task_scores, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MTEB aggregate: mean score across task categories"})


def cheatsheet():
    return "almteb: MTEB aggregate: mean score across task categories"
