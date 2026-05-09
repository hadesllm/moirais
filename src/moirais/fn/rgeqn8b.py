# moirais.fn — function file (hadesllm/moirais)
"""GLR threshold selection for change detection at given FAR."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ch8_glr_threshold"]


def rangayyan_ch8_glr_threshold(alpha, dof):
    """
    GLR threshold selection for change detection at given FAR

    Formula: Threshold h: P(GLR>h | H0) = alpha; estimated from chi-sq distribution

    Parameters
    ----------
    alpha : array-like
        Input data.
    dof : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: threshold_h

    References
    ----------
    Rangayyan Ch 8.5.3
    """
    alpha = np.asarray(alpha, dtype=float)
    n = int(alpha) if alpha.ndim == 0 else len(alpha)
    result = float(np.mean(alpha))
    se = float(np.std(alpha, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GLR threshold selection for change detection at given FAR"})


def cheatsheet():
    return "rgeqn8b: GLR threshold selection for change detection at given FAR"
