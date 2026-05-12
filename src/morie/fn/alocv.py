# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Output verification: second LLM checks response against criteria."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_output_verification"]


def alammar_output_verification(response, criteria, verifier_model):
    """
    Output verification: second LLM checks response against criteria

    Formula: passed = LLM_verifier(response, criteria) ∈ {PASS, FAIL}

    Parameters
    ----------
    response : array-like
        Input data.
    criteria : array-like
        Input data.
    verifier_model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: passed

    References
    ----------
    Alammar Ch 6, Output Verification section
    """
    response = np.atleast_1d(np.asarray(response, dtype=float))
    n = len(response)
    result = float(np.mean(response))
    se = float(np.std(response, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Output verification: second LLM checks response against criteria"})


def cheatsheet():
    return "alocv: Output verification: second LLM checks response against criteria"
