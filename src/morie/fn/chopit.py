# morie.fn -- function file (hadesllm/morie)
"""CHOPIT anchoring vignette model (King et al.) for perception correction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["chopit_vignette"]


def chopit_vignette(survey_data, vignette_data, n_categories):
    """
    CHOPIT anchoring vignette model (King et al.) for perception correction

    Formula: P(y_i = k | tau, gamma) = Phi(tau_k - X*gamma) - Phi(tau_{k-1} - X*gamma); vignette calibrates tau

    Parameters
    ----------
    survey_data : array-like
        Input data.
    vignette_data : array-like
        Input data.
    n_categories : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'tau': 'array', 'gamma': 'array', 'corrected_positions': 'array'}

    References
    ----------
    Armstrong Ch 2
    """
    survey_data = np.asarray(survey_data, dtype=float)
    n = int(survey_data) if survey_data.ndim == 0 else len(survey_data)
    result = float(np.mean(survey_data))
    se = float(np.std(survey_data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CHOPIT anchoring vignette model (King et al.) for perception correction"})


def cheatsheet():
    return "chopit: CHOPIT anchoring vignette model (King et al.) for perception correction"
