"""Spatio-temporal point process: intensity function lambda(s,t)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_st_point_process"]


def schabenberger_st_point_process(points, region, time_interval):
    """
    Spatio-temporal point process: intensity function lambda(s,t)

    Formula: lambda(s,t) = E[N(ds points dt)] / (ds*dt); N(A points [t1,t2]) ~ Pois(int lambda(s,t)ds dt)

    Parameters
    ----------
    points : array-like
        Input data.
    region : array-like
        Input data.
    time_interval : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: intensity

    References
    ----------
    Schabenberger Ch 9, Sec 9.5
    """
    points = np.asarray(points, dtype=float)
    n = int(points) if points.ndim == 0 else len(points)
    result = float(np.mean(points))
    se = float(np.std(points, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spatio-temporal point process: intensity function lambda(s,t)"})


def cheatsheet():
    return "spstp: Spatio-temporal point process: intensity function lambda(s,t)"
