"""Unfolding stress diagnostic. 'Shotgun!' -- Yusuke, Yu Yu Hakusho"""

from __future__ import annotations

from ._containers import DescriptiveResult


def unfolding_stress_diagnostic(X_resp, X_stim, observed):
    """Compute unfolding stress between ideal points and stimuli.

    Parameters
    ----------
    X_resp : array-like
        Respondent ideal points (n_resp x p).
    X_stim : array-like
        Stimulus coordinates (n_stim x p).
    observed : array-like
        Observed preference/distance matrix (n_resp x n_stim).

    Returns
    -------
    DescriptiveResult
        value = stress (float), extra has predicted distances.
    """
    import numpy as np

    Xr = np.asarray(X_resp, dtype=float)
    Xs = np.asarray(X_stim, dtype=float)
    O = np.asarray(observed, dtype=float)
    n_resp, n_stim = O.shape

    D_pred = np.zeros((n_resp, n_stim))
    for i in range(n_resp):
        for j in range(n_stim):
            D_pred[i, j] = np.sqrt(np.sum((Xr[i] - Xs[j]) ** 2))

    num = np.sum((O - D_pred) ** 2)
    denom = np.sum(O**2)
    stress = float(np.sqrt(num / denom)) if denom > 0 else 0.0
    return DescriptiveResult(
        name="unfolding_stress_diagnostic",
        value=stress,
        extra={"predicted": D_pred, "numerator": float(num)},
    )


unfst = unfolding_stress_diagnostic


def cheatsheet() -> str:
    return "unfolding_stress_diagnostic({}) -> Unfolding stress diagnostic. 'Shotgun!' -- Yusuke, Yu Yu Hak"
