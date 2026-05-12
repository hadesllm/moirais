"""Model stacking / super learner (OLS combiner). 'We are what they grow beyond.'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def stacking(
    X: np.ndarray,
    y: np.ndarray,
    base_predictions: list[np.ndarray],
) -> DescriptiveResult:
    """Stack multiple model predictions via OLS meta-learner.

    Parameters
    ----------
    X : ndarray
        Original features (unused -- kept for API consistency).
    y : ndarray
        True response.
    base_predictions : list[ndarray]
        Predictions from base learners, each shape (n,).

    Returns
    -------
    DescriptiveResult
    """
    y = np.asarray(y, dtype=float).ravel()
    Z = np.column_stack([np.ones(len(y))] + [np.asarray(p, dtype=float).ravel() for p in base_predictions])
    coeffs, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
    combined = Z @ coeffs
    residuals = y - combined
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return DescriptiveResult(
        name="Stacking (super learner)",
        value=r_squared,
        extra={
            "predictions": combined,
            "coefficients": coeffs,
            "r_squared": r_squared,
            "n_base": len(base_predictions),
            "n": len(y),
        },
    )


stck = stacking


def cheatsheet() -> str:
    return "stacking({}) -> Model stacking / super learner (OLS combiner). 'We are what "
