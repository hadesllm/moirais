"""Spatial data decomposition into trend + residuals."""

from __future__ import annotations

from ._containers import DescriptiveResult


def data_decomposition(Z, coords, trend_order=1):
    """Decompose spatial data into polynomial trend and residuals.

    Fits a polynomial surface of given order and returns trend + residuals.

    .. epigraph:: "No cost too great." -- Pale King, Hollow Knight

    Parameters
    ----------
    Z : array_like
        Observed values.
    coords : array_like
        Coordinates, shape ``(n, 2)``.
    trend_order : int
        Polynomial order (1=linear, 2=quadratic).

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np

    Z = np.asarray(Z, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)
    x, y = coords[:, 0], coords[:, 1]

    cols = [np.ones(len(Z))]
    if trend_order >= 1:
        cols.extend([x, y])
    if trend_order >= 2:
        cols.extend([x**2, x * y, y**2])
    if trend_order >= 3:
        cols.extend([x**3, x**2 * y, x * y**2, y**3])

    X = np.column_stack(cols)
    beta, res, rank, sv = np.linalg.lstsq(X, Z, rcond=None)
    trend = X @ beta
    residuals = Z - trend

    return DescriptiveResult(
        name="data_decomposition",
        value=float(np.var(residuals) / np.var(Z)) if np.var(Z) > 0 else 0.0,
        extra={
            "trend": trend,
            "residuals": residuals,
            "coefficients": beta.tolist(),
            "trend_order": trend_order,
            "trend_variance_ratio": float(np.var(trend) / np.var(Z)) if np.var(Z) > 0 else 0.0,
        },
    )


sgdcp = data_decomposition


def cheatsheet() -> str:
    return "data_decomposition({}) -> Spatial data decomposition into trend + residuals."
