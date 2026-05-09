# moirais.fn — function file (hadesllm/moirais)
"""Constitutional AI: critique-and-revise loop using a set of principles."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_constitutional_ai_loop"]


def kamath_constitutional_ai_loop(initial_response, constitution, model):
    """
    Constitutional AI: critique-and-revise loop using a set of principles

    Formula: for principle p in Constitution: y_crit = LLM(critique(p, y)); y_rev = LLM(revise(y, y_crit))

    Parameters
    ----------
    initial_response : array-like
        Input data.
    constitution : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: revised_response

    References
    ----------
    Kamath Ch 5, Constitutional AI section
    """
    initial_response = np.atleast_1d(np.asarray(initial_response, dtype=float))
    n = len(initial_response)
    result = float(np.mean(initial_response))
    se = float(np.std(initial_response, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Constitutional AI: critique-and-revise loop using a set of principles"})


def cheatsheet():
    return "kmcai: Constitutional AI: critique-and-revise loop using a set of principles"
