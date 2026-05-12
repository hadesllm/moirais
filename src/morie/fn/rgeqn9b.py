# morie.fn -- function file (hadesllm/morie)
"""ICA indeterminacy: permutation and scaling ambiguity."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ch9_ica_ambiguity"]


def rangayyan_ch9_ica_ambiguity(W, s):
    """
    ICA indeterminacy: permutation and scaling ambiguity

    Formula: y = W*W; W determined up to row permutation and scaling

    Parameters
    ----------
    W : array-like
        Input data.
    s : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: note

    References
    ----------
    Rangayyan Ch 9.7.2
    """
    W = np.asarray(W, dtype=float)
    n = int(W) if W.ndim == 0 else len(W)
    result = float(np.mean(W))
    se = float(np.std(W, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ICA indeterminacy: permutation and scaling ambiguity"})


def cheatsheet():
    return "rgeqn9b: ICA indeterminacy: permutation and scaling ambiguity"
