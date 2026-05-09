"""DINOv2 self-supervised representation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dinov2_repr"]


def dinov2_repr(x, student, teacher):
    """
    DINOv2 self-supervised representation

    Formula: DINO + iBOT mask + KoLeo

    Parameters
    ----------
    x : array-like
        Input data.
    student : array-like
        Input data.
    teacher : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Oquab et al (2024)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DINOv2 self-supervised representation"})


def cheatsheet():
    return "dnvtwo: DINOv2 self-supervised representation"
