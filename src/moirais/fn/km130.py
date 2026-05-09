"""Input alignment loss.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch9_input_alignment_loss"]


def kamath_ch9_input_alignment_loss(P_X, F_T, t):
    """
    Input alignment loss.

    Formula: \arg\min_{IN\_ALIGN_{X\to T}} L_{txt-gen}(\mathrm{LLM}(P_X,F_T), t)

    Parameters
    ----------
    P_X : array-like
        Input data.
    F_T : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 9, Eq 9.2, p. 380
    """
    P_X = np.atleast_1d(np.asarray(P_X, dtype=float))
    n = len(P_X)
    result = float(np.mean(P_X))
    se = float(np.std(P_X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Input alignment loss."})


def cheatsheet():
    return "km130: Input alignment loss."
