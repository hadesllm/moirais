# moirais.fn — function file (hadesllm/moirais)
"""Vibromyogram (VMG) signal characterization (lateral oscillation of contracting muscle)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_vmg"]


def rangayyan_vmg(vmg, fs):
    """It is during our darkest moments that we must focus to see the light. — Aristotle"""
    vmg = np.asarray(vmg, dtype=float)
    n = int(vmg) if vmg.ndim == 0 else len(vmg)
    result = float(np.mean(vmg))
    se = float(np.std(vmg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Vibromyogram (VMG) signal characterization (lateral oscillation of contracting muscle)"})


def cheatsheet():
    return "rgvmg: Vibromyogram (VMG) signal characterization (lateral oscillation of contracting muscle)"
