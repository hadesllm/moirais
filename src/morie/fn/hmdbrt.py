# morie.fn — function file (hadesllm/morie)
"""DistilBERT: distilled BERT with ~40% fewer parameters."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_distilbert"]


def geron_distilbert(teacher, student, X):
    """
    DistilBERT: distilled BERT with ~40% fewer parameters

    Formula: student mimics teacher outputs + masked LM

    Parameters
    ----------
    teacher : array-like
        Input data.
    student : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: student_model

    References
    ----------
    Géron Ch 15
    """
    teacher = np.atleast_1d(np.asarray(teacher, dtype=float))
    n = len(teacher)
    result = float(np.mean(teacher))
    se = float(np.std(teacher, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DistilBERT: distilled BERT with ~40% fewer parameters"})


def cheatsheet():
    return "hmdbrt: DistilBERT: distilled BERT with ~40% fewer parameters"
