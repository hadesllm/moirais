# moirais.fn — function file (hadesllm/moirais)
"""Knowing yourself is the beginning of all wisdom. — Aristotle"""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_isometric_contraction"]


def rangayyan_isometric_contraction(emg, force, fs):
    """Knowing yourself is the beginning of all wisdom. — Aristotle"""
    emg = np.asarray(emg, dtype=float)
    n = int(emg) if emg.ndim == 0 else len(emg)
    result = float(np.mean(emg))
    se = float(np.std(emg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Knowing yourself is the beginning of all wisdom. — Aristotle"})


def cheatsheet():
    return "Knowing yourself is the beginning of all wisdom. — Aristotle"
