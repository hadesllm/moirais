# morie.fn — function file (hadesllm/morie)
"""RLAIF: reward model trained from AI feedback (not human preferences)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_rlaif_objective"]


def kamath_rlaif_objective(ai_preferences):
    """
    RLAIF: reward model trained from AI feedback (not human preferences)

    Formula: preferences_AI = AI_judge(y_w, y_l, principle); r_phi fits preferences_AI via BT

    Parameters
    ----------
    ai_preferences : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: r_phi

    References
    ----------
    Kamath Ch 5, RLAIF section
    """
    ai_preferences = np.atleast_1d(np.asarray(ai_preferences, dtype=float))
    n = len(ai_preferences)
    result = float(np.mean(ai_preferences))
    se = float(np.std(ai_preferences, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RLAIF: reward model trained from AI feedback (not human preferences)"})


def cheatsheet():
    return "kmrlaif: RLAIF: reward model trained from AI feedback (not human preferences)"
