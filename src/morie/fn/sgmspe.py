"""Mean squared prediction error statistics."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def mspe_kriging(krig_var: np.ndarray) -> SpatialResult:
    r"""Summarize kriging variance (MSPE) over prediction locations.

    Parameters
    ----------
    krig_var : np.ndarray
        Kriging variances at prediction locations.

    Returns
    -------
    SpatialResult
        ``statistic`` is mean MSPE; ``extra`` has ``median``, ``max``,
        ``min``, ``std``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 5.

    .. epigraph::

    """
    kv = np.asarray(krig_var, dtype=np.float64).ravel()
    return SpatialResult(
        name="mspe_kriging",
        statistic=float(np.mean(kv)),
        p_value=None,
        extra={
            "median": float(np.median(kv)),
            "max": float(np.max(kv)),
            "min": float(np.min(kv)),
            "std": float(np.std(kv)),
        },
    )


sgmspe = mspe_kriging


def cheatsheet() -> str:
    return "mspe_kriging({}) -> Mean squared prediction error statistics."
