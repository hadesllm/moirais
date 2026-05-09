"""Space-time Moran scatterplot."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def spacetime_moran_scatter(
    values: np.ndarray,
    W: np.ndarray,
) -> SpatialResult:
    r"""
    Space-time Moran scatterplot analysis (Rey & Janikas, 2006).

    For each time period, computes the standardized variable *z* and
    its spatial lag *Wz*, then produces Moran's I as the slope of the
    scatterplot regression:

    .. math::

        I_t = \frac{\mathbf{z}_t^T \mathbf{W} \mathbf{z}_t}
                    {\mathbf{z}_t^T \mathbf{z}_t}

    Also classifies each unit-time into the four quadrants of the Moran
    scatterplot: HH (high-high), HL (high-low), LH (low-high), LL (low-low).

    Parameters
    ----------
    values : np.ndarray
        (n, T) panel with *n* spatial units and *T* time periods.
    W : np.ndarray
        (n, n) row-standardized spatial weight matrix.

    Returns
    -------
    SpatialResult
        statistic = time-averaged Moran's I, local_values = (T,) per-period I,
        extra has ``quadrants`` (n, T) array of int codes {0=HH,1=LH,2=LL,3=HL}.

    References
    ----------
    Rey SJ, Janikas MV (2006). STARS: Space-Time Analysis of Regional
    Systems. *Geographical Analysis*, 38(1), 67--86.
    doi:10.1111/j.0016-7363.2005.00675.x

    Anselin L (1995). Local indicators of spatial association -- LISA.
    *Geographical Analysis*, 27(2), 93--115.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n, T = 20, 5
    >>> W = np.ones((n, n)) / (n - 1)
    >>> np.fill_diagonal(W, 0)
    >>> vals = rng.normal(0, 1, (n, T))
    >>> res = spacetime_moran_scatter(vals, W)
    >>> len(res.local_values) == T
    True
    """
    values = np.asarray(values, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    n, T = values.shape
    if W.shape != (n, n):
        raise ValueError("W must be (n, n).")

    morans = np.empty(T)
    quadrants = np.empty((n, T), dtype=int)

    for tt in range(T):
        col = values[:, tt]
        mu = col.mean()
        sd = col.std(ddof=0)
        if sd < 1e-15:
            morans[tt] = 0.0
            quadrants[:, tt] = 2
            continue
        z = (col - mu) / sd
        Wz = W @ z
        morans[tt] = float(z @ Wz / (z @ z))

        for i in range(n):
            if z[i] >= 0 and Wz[i] >= 0:
                quadrants[i, tt] = 0  # HH
            elif z[i] < 0 and Wz[i] >= 0:
                quadrants[i, tt] = 1  # LH
            elif z[i] < 0 and Wz[i] < 0:
                quadrants[i, tt] = 2  # LL
            else:
                quadrants[i, tt] = 3  # HL

    return SpatialResult(
        name="spacetime_moran_scatter",
        statistic=float(np.mean(morans)),
        local_values=morans,
        extra={"quadrants": quadrants, "quadrant_labels": {0: "HH", 1: "LH", 2: "LL", 3: "HL"}, "n": n, "T": T},
    )


stmrn = spacetime_moran_scatter


def cheatsheet() -> str:
    return "spacetime_moran_scatter({}) -> Space-time Moran scatterplot."
