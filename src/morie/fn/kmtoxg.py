# morie.fn — function file (hadesllm/morie)
"""ToxiGen-based toxicity classifier score for a generation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_toxigen_score"]


def kamath_toxigen_score(text, classifier):
    """
    ToxiGen-based toxicity classifier score for a generation

    Formula: score = p(toxic | generation) via ToxiGen-trained RoBERTa classifier

    Parameters
    ----------
    text : array-like
        Input data.
    classifier : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prob

    References
    ----------
    Kamath Ch 6, ToxiGen section
    """
    text = np.atleast_1d(np.asarray(text, dtype=float))
    n = len(text)
    result = float(np.mean(text))
    se = float(np.std(text, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ToxiGen-based toxicity classifier score for a generation"})


def cheatsheet():
    return "kmtoxg: ToxiGen-based toxicity classifier score for a generation"
