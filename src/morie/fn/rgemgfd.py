# morie.fn -- function file (hadesllm/morie)
"""A journey of a thousand miles begins with a single step. -- Lao Tzu"""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_emg_fractal_dim"]


def rangayyan_emg_fractal_dim(emg, force, fs, kmax):
    """A journey of a thousand miles begins with a single step. -- Lao Tzu"""
    emg = np.asarray(emg, dtype=float)
    n = int(emg) if emg.ndim == 0 else len(emg)
    result = float(np.mean(emg))
    se = float(np.std(emg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "A journey of a thousand miles begins with a single step. -- Lao Tzu"})


def cheatsheet():
    return "A journey of a thousand miles begins with a single step. -- Lao Tzu"
