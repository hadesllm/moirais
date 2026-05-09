"""Unfolding residuals. 'Galick Gun!' -- Vegeta, Dragon Ball Z"""

from __future__ import annotations

from ._containers import DescriptiveResult


def unfolding_residuals(observed, predicted):
    """Compute residual matrix for unfolding model.

    Parameters
    ----------
    observed : array-like
        Observed distances/preferences.
    predicted : array-like
        Predicted distances from model.

    Returns
    -------
    DescriptiveResult
        value = residual matrix, extra has rmse.
    """
    import numpy as np

    O = np.asarray(observed, dtype=float)
    P = np.asarray(predicted, dtype=float)
    resid = O - P
    rmse = float(np.sqrt(np.mean(resid**2)))
    return DescriptiveResult(
        name="unfolding_residuals", value=resid, extra={"rmse": rmse, "mean_resid": float(np.mean(resid))}
    )


ufres = unfolding_residuals


def cheatsheet() -> str:
    return "unfolding_residuals({}) -> Unfolding residuals. 'Galick Gun!' -- Vegeta, Dragon Ball Z"
