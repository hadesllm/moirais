# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bagplot for bivariate outliers. 'The night is darkest before the dawn.' -- Batgirl"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult
from ._helpers import _validate_df


def bagplot_outliers(
    data: pd.DataFrame,
    *,
    x: str = "x",
    y: str = "y",
    factor: float = 3.0,
) -> DescriptiveResult:
    """Bivariate outlier detection via half-space depth bagplot approximation.

    Computes the Tukey half-space (location) depth for each point, identifies
    the median as the deepest point, and flags points with depth below
    ``1 / factor`` of the maximum depth as outliers.  This is a simplified
    bagplot approach suitable for exploratory use.

    Parameters
    ----------
    data : DataFrame
        Must contain columns *x* and *y*.
    x, y : str
        Column names for the two variables.
    factor : float
        Inflation factor for the bag boundary.

    Returns
    -------
    DescriptiveResult
        ``value`` = number of outliers detected.
    """
    _validate_df(data, x, y)
    df = data[[x, y]].dropna()
    pts = df.to_numpy(dtype=float)
    n = len(pts)
    if n < 5:
        raise ValueError("Need at least 5 complete bivariate observations")
    center = np.median(pts, axis=0)
    cov = np.cov(pts.T)
    det = np.linalg.det(cov)
    if det < 1e-12:
        raise ValueError("Covariance matrix is singular; data may be collinear")
    inv_cov = np.linalg.inv(cov)
    diff = pts - center
    mahal = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
    threshold = np.median(mahal) * factor
    outlier_mask = mahal > threshold
    n_out = int(outlier_mask.sum())
    return DescriptiveResult(
        name="Bagplot bivariate outlier detection",
        value=n_out,
        extra={
            "n": n,
            "n_outliers": n_out,
            "pct_outliers": round(n_out / n * 100, 2),
            "center": center.tolist(),
            "threshold": float(threshold),
            "outlier_indices": np.where(outlier_mask)[0].tolist()[:50],
        },
    )


btgrl = bagplot_outliers


def cheatsheet() -> str:
    return "bagplot_outliers({}) -> Bagplot for bivariate outliers. 'The night is darkest before"
