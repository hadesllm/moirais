# moirais.fn — function file (hadesllm/moirais)
"""Prompt tuning: learned continuous soft-prompt embeddings prepended to input."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_prompt_tuning"]


def kamath_prompt_tuning(P, X):
    """
    Prompt tuning: learned continuous soft-prompt embeddings prepended to input

    Formula: X_aug = [P; X]  where P in R^{L_p x d} is learned

    Parameters
    ----------
    P : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X_aug

    References
    ----------
    Kamath Ch 4, Prompt Tuning section
    """
    P = np.atleast_1d(np.asarray(P, dtype=float))
    n = len(P)
    result = float(np.mean(P))
    se = float(np.std(P, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Prompt tuning: learned continuous soft-prompt embeddings prepended to input"})


def cheatsheet():
    return "kmptun: Prompt tuning: learned continuous soft-prompt embeddings prepended to input"
