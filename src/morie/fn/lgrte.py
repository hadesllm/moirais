# morie.fn — function file (hadesllm/morie)
"""Local growth rate estimation across spatial units over time."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def local_growth_rate(
    values: np.ndarray,
    method: str = "log_ratio",
) -> SpatialResult:
    r"""
    Estimate local growth rates for spatial units across time periods.

    For each spatial unit *i* and consecutive time periods *t* and *t+1*,
    computes the growth rate using one of two methods:

    **Log-ratio** (default):

    .. math::

        g_{i,t} = \ln\!\left(\frac{y_{i,t+1}}{y_{i,t}}\right)

    **Percentage change**:

    .. math::

        g_{i,t} = \frac{y_{i,t+1} - y_{i,t}}{y_{i,t}}

    Parameters
    ----------
    values : np.ndarray
        (n_units, n_times) matrix of values over time for each spatial unit.
    method : str
        ``"log_ratio"`` or ``"pct_change"`` (default ``"log_ratio"``).

    Returns
    -------
    SpatialResult
        statistic = overall mean growth rate, local_values = (n_units,)
        mean growth per unit, extra has ``growth_matrix`` of shape
        (n_units, n_times - 1).

    References
    ----------
    Rey SJ, Janikas MV (2006). STARS: Space-Time Analysis of Regional
    Systems. *Geographical Analysis*, 38(1), 67--86.
    doi:10.1111/j.0016-7363.2005.00675.x

    Anselin L, Rey SJ (2014). *Modern Spatial Econometrics in Practice*.
    GeoDa Press, Chicago. Chapter 12.

    Examples
    --------
    >>> import numpy as np
    >>> vals = np.array([[100, 110, 121], [200, 190, 200]], dtype=float)
    >>> res = local_growth_rate(vals)
    >>> res.local_values.shape == (2,)
    True
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("values must be (n_units, n_times).")
    n_units, n_times = values.shape
    if n_times < 2:
        raise ValueError("Need at least 2 time periods.")

    if method == "log_ratio":
        safe_vals = np.where(values <= 0, np.nan, values)
        growth = np.log(safe_vals[:, 1:]) - np.log(safe_vals[:, :-1])
    elif method == "pct_change":
        denom = np.where(values[:, :-1] == 0, np.nan, values[:, :-1])
        growth = (values[:, 1:] - values[:, :-1]) / denom
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'log_ratio' or 'pct_change'.")

    unit_means = np.nanmean(growth, axis=1)
    overall = float(np.nanmean(unit_means))

    return SpatialResult(
        name="local_growth_rate",
        statistic=overall,
        local_values=unit_means,
        extra={"growth_matrix": growth, "method": method, "n_units": n_units, "n_times": n_times},
    )


lgrte = local_growth_rate


def cheatsheet() -> str:
    return "local_growth_rate({}) -> Local growth rate estimation across spatial units over time."
