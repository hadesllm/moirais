# moirais.fn — function file (hadesllm/moirais)
"""Ordinary kriging interpolation."""

import numpy as np
from scipy.spatial.distance import cdist

from ._containers import DescriptiveResult


def _spherical_variogram(h: np.ndarray, nugget: float, sill: float, range_: float) -> np.ndarray:
    """Spherical variogram model."""
    h = np.asarray(h, dtype=float)
    gamma = np.where(
        h == 0,
        0.0,
        np.where(
            h <= range_,
            nugget + (sill - nugget) * (1.5 * h / range_ - 0.5 * (h / range_) ** 3),
            sill,
        ),
    )
    return gamma


def ordinary_kriging(
    known_coords: np.ndarray,
    known_values: np.ndarray,
    predict_coords: np.ndarray,
    *,
    nugget: float = 0.0,
    sill: float = 1.0,
    range_: float = 1.0,
) -> DescriptiveResult:
    """
    Ordinary kriging using a spherical variogram model.

    Solves the kriging system for each prediction point:

    .. math::

        \\begin{bmatrix} \\boldsymbol{\\Gamma} & \\mathbf{1} \\\\
        \\mathbf{1}^T & 0 \\end{bmatrix}
        \\begin{bmatrix} \\boldsymbol{\\lambda} \\\\ \\mu \\end{bmatrix}
        = \\begin{bmatrix} \\boldsymbol{\\gamma}_0 \\\\ 1 \\end{bmatrix}

    where :math:`\\Gamma_{ij} = \\gamma(h_{ij})` is the semivariance between
    known points, and :math:`\\gamma_0` is the semivariance between known
    points and the prediction location.

    :param known_coords: Coordinates of observed locations, shape (n, 2).
    :param known_values: Observed values, shape (n,).
    :param predict_coords: Coordinates where predictions are desired, shape (m, 2).
    :param nugget: Variogram nugget (discontinuity at origin). Default 0.0.
    :param sill: Variogram sill (total variance). Default 1.0.
    :param range_: Variogram range (distance at which sill is reached). Default 1.0.
    :return: :class:`DescriptiveResult` with value = predicted values array (m,),
        extra contains kriging variances.
    :raises ValueError: If array shapes are incompatible.

    References
    ----------
    Cressie, N. A. C. (1993). *Statistics for Spatial Data* (rev. ed.). Wiley.

    Goovaerts, P. (1997). *Geostatistics for Natural Resources Evaluation*.
    Oxford Univ. Press.
    """
    known_coords = np.asarray(known_coords, dtype=float)
    known_values = np.asarray(known_values, dtype=float)
    predict_coords = np.asarray(predict_coords, dtype=float)
    n = len(known_values)
    m = predict_coords.shape[0]

    if known_coords.shape[0] != n:
        raise ValueError("known_coords and known_values must have same length.")

    dist_known = cdist(known_coords, known_coords)
    Gamma = _spherical_variogram(dist_known, nugget, sill, range_)

    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = Gamma
    A[:n, n] = 1.0
    A[n, :n] = 1.0

    dist_pred = cdist(known_coords, predict_coords)

    predictions = np.zeros(m)
    variances = np.zeros(m)

    for j in range(m):
        gamma_0 = _spherical_variogram(dist_pred[:, j], nugget, sill, range_)
        b = np.zeros(n + 1)
        b[:n] = gamma_0
        b[n] = 1.0

        try:
            lam = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            lam = np.linalg.lstsq(A, b, rcond=None)[0]

        predictions[j] = float(lam[:n] @ known_values)
        variances[j] = float(lam[:n] @ gamma_0 + lam[n])

    return DescriptiveResult(
        name="Ordinary Kriging",
        value=predictions,
        extra={"variances": variances, "nugget": nugget, "sill": sill, "range": range_, "n_known": n, "n_predict": m},
    )


krig_fn = ordinary_kriging


def cheatsheet() -> str:
    return "_spherical_variogram({}) -> Ordinary kriging interpolation."
