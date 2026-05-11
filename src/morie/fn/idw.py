# morie.fn — function file (hadesllm/morie)
"""Inverse distance weighting interpolation."""

import numpy as np
from scipy.spatial.distance import cdist

from ._containers import DescriptiveResult


def inverse_distance_weighting(
    known_coords: np.ndarray,
    known_values: np.ndarray,
    predict_coords: np.ndarray,
    *,
    power: float = 2.0,
) -> DescriptiveResult:
    """
    Inverse distance weighting (IDW) spatial interpolation.

    For each prediction point :math:`s_0`:

    .. math::

        \\hat{z}(s_0) = \\frac{\\sum_{i=1}^{n} w_i \\, z(s_i)}
                              {\\sum_{i=1}^{n} w_i}, \\qquad
        w_i = \\frac{1}{d(s_0, s_i)^p}

    If :math:`s_0` coincides with a known point, the known value is returned
    exactly (no division by zero).

    :param known_coords: Coordinates of observed locations, shape (n, 2).
    :param known_values: Observed values, shape (n,).
    :param predict_coords: Coordinates for prediction, shape (m, 2).
    :param power: Distance exponent *p*. Default 2.0 (inverse-square).
    :return: :class:`DescriptiveResult` with value = predicted values array (m,).
    :raises ValueError: If shapes are incompatible or power < 0.

    References
    ----------
    Shepard, D. (1968). A two-dimensional interpolation function for
    irregularly-spaced data. *Proc. 23rd ACM National Conference*, 517-524.
    https://doi.org/10.1145/800186.810616
    """
    known_coords = np.asarray(known_coords, dtype=float)
    known_values = np.asarray(known_values, dtype=float)
    predict_coords = np.asarray(predict_coords, dtype=float)

    if power < 0:
        raise ValueError(f"power must be >= 0, got {power}.")
    n = len(known_values)
    if known_coords.shape[0] != n:
        raise ValueError("known_coords and known_values must have same length.")

    dists = cdist(predict_coords, known_coords)

    predictions = np.zeros(predict_coords.shape[0])

    for j in range(len(predictions)):
        d = dists[j]
        zero_mask = d == 0.0
        if np.any(zero_mask):
            idx = np.argmax(zero_mask)
            predictions[j] = known_values[idx]
        else:
            w = 1.0 / d**power
            predictions[j] = float(np.dot(w, known_values) / w.sum())

    return DescriptiveResult(
        name="IDW Interpolation",
        value=predictions,
        extra={"power": power, "n_known": n, "n_predict": len(predictions)},
    )


idw_fn = inverse_distance_weighting


def cheatsheet() -> str:
    return "inverse_distance_weighting({}) -> Inverse distance weighting interpolation."
