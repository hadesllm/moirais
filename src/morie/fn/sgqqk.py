"""QQ plot data for kriging standardized errors."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def qq_plot_kriging(standardized_errors: np.ndarray) -> SpatialResult:
    r"""Generate QQ-plot data for standardized kriging errors.

    Parameters
    ----------
    standardized_errors : np.ndarray
        Standardized prediction errors.

    Returns
    -------
    SpatialResult
        ``statistic`` is the correlation between theoretical and
        sample quantiles.
        ``extra`` has ``theoretical``, ``sample``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 5.

    .. epigraph::

        "I need a weapon." -- Master Chief, Halo
    """
    from scipy.stats import norm

    se = np.sort(np.asarray(standardized_errors, dtype=np.float64).ravel())
    n = len(se)
    theoretical = norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    corr = float(np.corrcoef(theoretical, se)[0, 1]) if n > 1 else 1.0

    return SpatialResult(
        name="qq_plot_kriging",
        statistic=corr,
        p_value=None,
        extra={"theoretical": theoretical, "sample": se},
    )


sgqqk = qq_plot_kriging


def cheatsheet() -> str:
    return "qq_plot_kriging({}) -> QQ plot data for kriging standardized errors."
