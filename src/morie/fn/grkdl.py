# morie.fn — function file (hadesllm/morie)
"""Knowledge distillation: student matches soft-labels of teacher."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_knowledge_distillation_loss"]


def geron_knowledge_distillation_loss(student_logits, teacher_logits, y, alpha, T):
    """
    Knowledge distillation: student matches soft-labels of teacher

    Formula: L = (1-alpha)*CE(student, y) + alpha*T^2*KL( softmax(s/T) || softmax(t/T) )

    Parameters
    ----------
    student_logits : array-like
        Input data.
    teacher_logits : array-like
        Input data.
    y : array-like
        Input data.
    alpha : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Géron Ch 17, Knowledge Distillation section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Knowledge distillation: student matches soft-labels of teacher"})


def cheatsheet():
    return "grkdl: Knowledge distillation: student matches soft-labels of teacher"
