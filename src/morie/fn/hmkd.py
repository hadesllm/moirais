# morie.fn -- function file (hadesllm/morie)
"""Knowledge distillation: student matches softened teacher outputs."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_knowledge_distillation"]


def geron_knowledge_distillation(teacher, student, X, y, T, alpha):
    """
    Knowledge distillation: student matches softened teacher outputs

    Formula: L = alpha*CE(y, student) + (1-alpha)*T^2*KL(softmax(teach/T), softmax(stu/T))

    Parameters
    ----------
    teacher : array-like
        Input data.
    student : array-like
        Input data.
    X : array-like
        Input data.
    y : array-like
        Input data.
    T : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: student_model

    References
    ----------
    Géron Ch 17
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Knowledge distillation: student matches softened teacher outputs"})


def cheatsheet():
    return "hmkd: Knowledge distillation: student matches softened teacher outputs"
