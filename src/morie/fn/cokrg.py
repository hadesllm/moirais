# morie.fn — function file (hadesllm/morie)
"""Co-kriging — multivariate spatial prediction (Schabenberger & Gotway Ch 5)."""

import numpy as np
from scipy.spatial.distance import cdist


def cokrg(
    coords: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
    target: np.ndarray,
    *,
    sill_p: float = 1.0,
    range_p: float = 1.0,
    sill_s: float = 1.0,
    range_s: float = 1.0,
    cross_sill: float = 0.5,
    cross_range: float = 1.0,
    nugget: float = 0.0,
) -> dict:
    """
    Simple co-kriging using a primary and secondary variable.

    Builds the full co-kriging system with auto- and cross-covariance
    matrices to predict the primary variable at target locations.

    :param coords: Observation coordinates (n, 2) — same for both variables.
    :param primary: Primary variable values (n,).
    :param secondary: Secondary (covariate) values (n,).
    :param target: Prediction coordinates (m, 2).
    :param sill_p: Sill of primary auto-covariance.
    :param range_p: Range of primary auto-covariance.
    :param sill_s: Sill of secondary auto-covariance.
    :param range_s: Range of secondary auto-covariance.
    :param cross_sill: Sill of cross-covariance.
    :param cross_range: Range of cross-covariance.
    :param nugget: Nugget variance.
    :return: dict with ``predictions``, ``variances``.
    :raises ValueError: If shapes are incompatible.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 5.
    """
    coords = np.asarray(coords, dtype=float)
    primary = np.asarray(primary, dtype=float)
    secondary = np.asarray(secondary, dtype=float)
    target = np.asarray(target, dtype=float)
    n = len(primary)
    if coords.shape != (n, 2):
        raise ValueError(f"coords must be ({n}, 2), got {coords.shape}.")
    if len(secondary) != n:
        raise ValueError("secondary must have same length as primary.")
    if target.ndim == 1:
        target = target.reshape(1, -1)

    D = cdist(coords, coords)
    Cpp = sill_p * np.exp(-D / (range_p + 1e-12))
    Css = sill_s * np.exp(-D / (range_s + 1e-12))
    Cps = cross_sill * np.exp(-D / (cross_range + 1e-12))

    C = np.block([[Cpp + nugget * np.eye(n), Cps], [Cps.T, Css + nugget * np.eye(n)]])

    z = np.concatenate([primary, secondary])
    m = len(target)
    preds = np.zeros(m)
    variances = np.zeros(m)

    for j in range(m):
        d0 = cdist(coords, target[j:j + 1]).ravel()
        c0p = sill_p * np.exp(-d0 / (range_p + 1e-12))
        c0s = cross_sill * np.exp(-d0 / (cross_range + 1e-12))
        c0 = np.concatenate([c0p, c0s])

        try:
            w = np.linalg.solve(C, c0)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(C, c0, rcond=None)[0]
        preds[j] = float(w @ z)
        variances[j] = max(float(sill_p + nugget - w @ c0), 0.0)

    return {
        "predictions": preds,
        "variances": variances,
        "n": n,
        "n_target": m,
    }


cokrg_fn = cokrg


def cheatsheet() -> str:
    return "cokrg({}) -> Co-kriging multivariate spatial prediction."
