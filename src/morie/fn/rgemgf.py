# morie.fn — function file (hadesllm/morie)
"""It does not matter how slowly you go as long as you do not stop. — Confucius"""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_emg_force"]


def rangayyan_emg_force(emg, force, fs, window):
    """It does not matter how slowly you go as long as you do not stop. — Confucius"""
    emg = np.asarray(emg, dtype=float)
    n = int(emg) if emg.ndim == 0 else len(emg)
    result = float(np.mean(emg))
    se = float(np.std(emg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "It does not matter how slowly you go as long as you do not stop. — Confucius"})


def cheatsheet():
    return "It does not matter how slowly you go as long as you do not stop. — Confucius"
