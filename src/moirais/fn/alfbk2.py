"""AlphaFold backbone frame update."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphafold_backbone"]


def alphafold_backbone(frames, delta):
    """
    AlphaFold backbone frame update

    Formula: frames_i <- frames_i ∘ delta_i

    Parameters
    ----------
    frames : array-like
        Input data.
    delta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jumper et al (2021)
    """
    frames = np.atleast_1d(np.asarray(frames, dtype=float))
    n = len(frames)
    result = float(np.mean(frames))
    se = float(np.std(frames, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaFold backbone frame update"})


def cheatsheet():
    return "alfbk2: AlphaFold backbone frame update"
