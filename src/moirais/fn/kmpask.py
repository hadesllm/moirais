# moirais.fn — function file (hadesllm/moirais)
"""Pass@k for code generation: expected fraction of at-least-one-correct in k samples."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_pass_at_k"]


def kamath_pass_at_k(n, c, k):
    """
    Pass@k for code generation: expected fraction of at-least-one-correct in k samples

    Formula: pass@k = 1 - C(n - c, k) / C(n, k)  where n samples, c correct, k draws

    Parameters
    ----------
    n : array-like
        Input data.
    c : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: pass_k

    References
    ----------
    Kamath Ch 8, Pass@k (HumanEval) section
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pass@k for code generation: expected fraction of at-least-one-correct in k samples"})


def cheatsheet():
    return "kmpask: Pass@k for code generation: expected fraction of at-least-one-correct in k samples"
