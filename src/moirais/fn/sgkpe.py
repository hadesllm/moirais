"""Kriging prediction error statistics."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def kriging_prediction_error(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> SpatialResult:
    r"""Compute prediction error diagnostics.

    Parameters
    ----------
    observed : np.ndarray
        True values.
    predicted : np.ndarray
        Kriging predictions.

    Returns
    -------
    SpatialResult
        ``statistic`` is RMSE; ``extra`` has ``mae``, ``me``, ``mse``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 5.

    .. epigraph::

        "I should go." -- Shepard, Mass Effect
    """
    obs = np.asarray(observed, dtype=np.float64).ravel()
    pred = np.asarray(predicted, dtype=np.float64).ravel()
    err = obs - pred
    me = float(np.mean(err))
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    return SpatialResult(
        name="kriging_prediction_error",
        statistic=rmse,
        p_value=None,
        extra={"me": me, "mse": mse, "mae": mae, "errors": err},
    )


sgkpe = kriging_prediction_error


def cheatsheet() -> str:
    return "kriging_prediction_error({}) -> Kriging prediction error statistics."
