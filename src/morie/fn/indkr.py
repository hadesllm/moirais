# morie.fn — function file (hadesllm/morie)
"""Indicator kriging — binary outcome (Schabenberger & Gotway Ch 5)."""

import numpy as np
from scipy.spatial.distance import cdist


def indkr(
    coords: np.ndarray,
    values: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float | None = None,
    sill: float = 0.25,
    range_param: float = 1.0,
    nugget: float = 0.0,
) -> dict:
    """
    Indicator kriging for exceedance probability mapping.

    Transforms values to indicators (1 if above threshold, else 0)
    and applies ordinary kriging to predict the probability of
    exceedance at target locations.

    :param coords: Observation coordinates (n, 2).
    :param values: Observed values (n,).
    :param target: Prediction coordinates (m, 2).
    :param threshold: Threshold for the indicator. Defaults to the median.
    :param sill: Variogram sill for the indicator.
    :param range_param: Variogram range.
    :param nugget: Nugget variance.
    :return: dict with ``probabilities``, ``threshold``, ``indicators``.
    :raises ValueError: If shapes are incompatible.

    References
    ----------
    Journel, A. G. (1983). Nonparametric estimation of spatial
    distributions. *Math. Geology*, 15(3), 445-468.

    Schabenberger & Gotway (2005), Ch. 5.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    target = np.asarray(target, dtype=float)
    n = len(values)
    if coords.shape != (n, 2):
        raise ValueError(f"coords must be ({n}, 2), got {coords.shape}.")
    if target.ndim == 1:
        target = target.reshape(1, -1)

    if threshold is None:
        threshold = float(np.median(values))

    indicators = (values > threshold).astype(float)

    D = cdist(coords, coords)
    C = sill * np.exp(-D / (range_param + 1e-12))
    np.fill_diagonal(C, C.diagonal() + nugget)

    C_aug = np.zeros((n + 1, n + 1))
    C_aug[:n, :n] = C
    C_aug[:n, n] = 1.0
    C_aug[n, :n] = 1.0

    m = len(target)
    probs = np.zeros(m)
    for j in range(m):
        d0 = cdist(coords, target[j:j + 1]).ravel()
        c0 = sill * np.exp(-d0 / (range_param + 1e-12))
        rhs = np.append(c0, 1.0)
        try:
            lam = np.linalg.solve(C_aug, rhs)
        except np.linalg.LinAlgError:
            lam = np.linalg.lstsq(C_aug, rhs, rcond=None)[0]
        p = float(lam[:n] @ indicators)
        probs[j] = np.clip(p, 0.0, 1.0)

    return {
        "probabilities": probs,
        "threshold": threshold,
        "indicators": indicators,
        "n": n,
        "n_target": m,
    }


indkr_fn = indkr


def cheatsheet() -> str:
    return "indkr({}) -> Indicator kriging for exceedance probability."
