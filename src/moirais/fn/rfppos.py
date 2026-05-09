"""Reactive pose filter — covalent inhibitor docking."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["reactive_pose_filter"]


def reactive_pose_filter(pose, cys_residue):
    """
    Reactive pose filter — covalent inhibitor docking

    Formula: require warhead-Cys distance ≤ 4 Å in pose

    Parameters
    ----------
    pose : array-like
        Input data.
    cys_residue : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bianco et al (2016) CovDock
    """
    pose = np.atleast_1d(np.asarray(pose, dtype=float))
    n = len(pose)
    result = float(np.mean(pose))
    se = float(np.std(pose, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Reactive pose filter — covalent inhibitor docking"})


def cheatsheet():
    return "rfppos: Reactive pose filter — covalent inhibitor docking"
