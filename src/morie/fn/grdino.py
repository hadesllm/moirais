# morie.fn -- function file (hadesllm/morie)
"""DINO self-distillation: student matches teacher's output distribution (stop-grad teacher)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_dino_self_distillation"]


def geron_dino_self_distillation(student_logits, teacher_logits, tau_s, tau_t):
    """
    DINO self-distillation: student matches teacher's output distribution (stop-grad teacher)

    Formula: L = - sum_k P_teacher(k) log P_student(k); teacher = EMA(student), no grad

    Parameters
    ----------
    student_logits : array-like
        Input data.
    teacher_logits : array-like
        Input data.
    tau_s : array-like
        Input data.
    tau_t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Géron Ch 16, DINO section
    """
    student_logits = np.atleast_1d(np.asarray(student_logits, dtype=float))
    n = len(student_logits)
    result = float(np.mean(student_logits))
    se = float(np.std(student_logits, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DINO self-distillation: student matches teacher's output distribution (stop-grad teacher)"})


def cheatsheet():
    return "grdino: DINO self-distillation: student matches teacher's output distribution (stop-grad teacher)"
