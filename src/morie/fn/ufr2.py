"""Unfolding R-squared. 'Super Saiyan Blue.' -- Goku, Dragon Ball Z"""

from __future__ import annotations

from ._containers import DescriptiveResult


def unfolding_r_squared(observed, predicted):
    """Compute R-squared for an unfolding model.

    Parameters
    ----------
    observed : array-like
        Observed distances/preferences.
    predicted : array-like
        Predicted distances from model.

    Returns
    -------
    DescriptiveResult
        value = R-squared (float).
    """
    import numpy as np

    O = np.asarray(observed, dtype=float).ravel()
    P = np.asarray(predicted, dtype=float).ravel()
    ss_res = np.sum((O - P) ** 2)
    ss_tot = np.sum((O - np.mean(O)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return DescriptiveResult(
        name="unfolding_r_squared", value=float(r2), extra={"ss_res": float(ss_res), "ss_tot": float(ss_tot)}
    )


ufr2 = unfolding_r_squared


def cheatsheet() -> str:
    return "unfolding_r_squared({}) -> Unfolding R-squared. 'Super Saiyan Blue.' -- Goku, Dragon Ba"
