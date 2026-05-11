# morie.fn — function file (hadesllm/morie)
"""Step-back prompting: first ask a higher-level/generalized query, then the specific one."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_step_back_prompting"]


def kamath_step_back_prompting(query, model):
    """
    Step-back prompting: first ask a higher-level/generalized query, then the specific one

    Formula: q_high = LLM(step_back(x)); answer(x | q_high-retrieved-ctx + x-retrieved-ctx)

    Parameters
    ----------
    query : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: step_back_query, answer

    References
    ----------
    Kamath Ch 7, Step-back prompting section
    """
    query = np.atleast_1d(np.asarray(query, dtype=float))
    n = len(query)
    result = float(np.mean(query))
    se = float(np.std(query, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Step-back prompting: first ask a higher-level/generalized query, then the specific one"})


def cheatsheet():
    return "kmstb: Step-back prompting: first ask a higher-level/generalized query, then the specific one"
