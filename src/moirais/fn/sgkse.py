"""Standardized kriging prediction errors."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def standardized_prediction_error(
    errors: np.ndarray,
    krig_var: np.ndarray,
) -> SpatialResult:
    r"""Standardize kriging errors by their predicted variance.

    .. math::

        e_i^* = \frac{Z(s_i) - \hat{Z}(s_i)}{\sigma_{OK}(s_i)}

    Parameters
    ----------
    errors : np.ndarray
        Raw prediction errors.
    krig_var : np.ndarray
        Kriging variances (must be positive).

    Returns
    -------
    SpatialResult
        ``statistic`` is mean of squared standardized errors (ideally 1).
        ``extra`` has ``std_errors``, ``mean_std_error``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 5.

    .. epigraph::

        "We'll bang, ok?" -- Shepard, Mass Effect
    """
    err = np.asarray(errors, dtype=np.float64).ravel()
    kv = np.asarray(krig_var, dtype=np.float64).ravel()
    kv = np.maximum(kv, 1e-10)
    std_err = err / np.sqrt(kv)
    msse = float(np.mean(std_err**2))

    return SpatialResult(
        name="standardized_prediction_error",
        statistic=msse,
        p_value=None,
        extra={
            "std_errors": std_err,
            "mean_std_error": float(np.mean(std_err)),
        },
    )


sgkse = standardized_prediction_error


def cheatsheet() -> str:
    return "standardized_prediction_error({}) -> Standardized kriging prediction errors."
