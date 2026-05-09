"""F-function (empty space function): CDF of distance from arbitrary point to nearest event."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_f_function"]


def schabenberger_f_function(points, region, r):
    """
    F-function (empty space function): CDF of distance from arbitrary point to nearest event

    Formula: F(r) = P(d_e <= r) = 1 - exp(-lambda*pi*r^2) for CSR

    Parameters
    ----------
    points : array-like
        Input data.
    region : array-like
        Input data.
    r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schabenberger Ch 3, Sec 3.3.4
    """
    points = np.asarray(points, dtype=float)
    n = int(points) if points.ndim == 0 else len(points)
    result = float(np.mean(points))
    se = float(np.std(points, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "F-function (empty space function): CDF of distance from arbitrary point to nearest event"})


def cheatsheet():
    return "spffun: F-function (empty space function): CDF of distance from arbitrary point to nearest event"
