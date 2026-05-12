# morie.fn -- function file (hadesllm/morie)
"""Knee-joint sound generation model (patellofemoral crepitus)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_knee_joint_sound"]


def rangayyan_knee_joint_sound(vag, fs, force):
    """In God we trust; all others must bring data. -- W. Edwards Deming"""
    vag = np.asarray(vag, dtype=float)
    n = int(vag) if vag.ndim == 0 else len(vag)
    result = float(np.mean(vag))
    se = float(np.std(vag, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Knee-joint sound generation model (patellofemoral crepitus)"})


def cheatsheet():
    return "rgkneejt: Knee-joint sound generation model (patellofemoral crepitus)"
