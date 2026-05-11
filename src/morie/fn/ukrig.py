"""Universal kriging with trend (Schabenberger & Gotway Ch 5)."""

import numpy as np
from scipy.spatial.distance import cdist


def ukrig(
    coords: np.ndarray,
    values: np.ndarray,
    target: np.ndarray,
    *,
    sill: float = 1.0,
    range_param: float = 1.0,
    nugget: float = 0.0,
    trend_order: int = 1,
    model: str = "exponential",
) -> dict:
    """
    Universal kriging with polynomial trend.

    Extends ordinary kriging by including a polynomial drift function
    in the mean structure, estimated jointly with the kriging weights.

    :param coords: Training coordinates (n, 2).
    :param values: Training values (n,).
    :param target: Prediction coordinates (m, 2).
    :param sill: Variogram sill.
    :param range_param: Variogram range.
    :param nugget: Nugget variance.
    :param trend_order: Order of polynomial trend (0, 1, or 2).
    :param model: ``'exponential'`` or ``'gaussian'``.
    :return: dict with ``predictions``, ``variances``, ``weights``.
    :raises ValueError: If shapes or model are invalid.

    References
    ----------
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

    def _cov(d):
        if model == "exponential":
            return sill * np.exp(-d / (range_param + 1e-12))
        elif model == "gaussian":
            return sill * np.exp(-d ** 2 / (range_param ** 2 + 1e-12))
        raise ValueError(f"Unknown model: {model}")

    def _trend_matrix(c):
        ones = np.ones((len(c), 1))
        if trend_order == 0:
            return ones
        elif trend_order == 1:
            return np.hstack([ones, c])
        else:
            return np.hstack([ones, c, (c ** 2), (c[:, 0:1] * c[:, 1:2])])

    C = _cov(cdist(coords, coords))
    np.fill_diagonal(C, C.diagonal() + nugget)
    F = _trend_matrix(coords)
    p = F.shape[1]

    K = np.zeros((n + p, n + p))
    K[:n, :n] = C
    K[:n, n:] = F
    K[n:, :n] = F.T

    m = len(target)
    preds = np.zeros(m)
    variances = np.zeros(m)
    all_weights = np.zeros((m, n))

    for j in range(m):
        c0 = _cov(cdist(coords, target[j:j + 1]).ravel())
        f0 = _trend_matrix(target[j:j + 1]).ravel()
        rhs = np.concatenate([c0, f0])
        try:
            lam = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            lam = np.linalg.lstsq(K, rhs, rcond=None)[0]
        w = lam[:n]
        preds[j] = float(w @ values)
        variances[j] = max(float(sill + nugget - lam @ rhs), 0.0)
        all_weights[j] = w

    return {
        "predictions": preds,
        "variances": variances,
        "weights": all_weights,
        "trend_order": trend_order,
        "model": model,
        "n": n,
    }


ukrig_fn = ukrig


def cheatsheet() -> str:
    return "ukrig({}) -> Universal kriging with polynomial trend."
