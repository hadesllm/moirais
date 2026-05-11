"""Separable spatio-temporal covariance (Schabenberger & Gotway Ch 9)."""

import numpy as np
from scipy.spatial.distance import cdist


def stsep(
    coords: np.ndarray,
    times: np.ndarray,
    *,
    spatial_sill: float = 1.0,
    spatial_range: float = 1.0,
    temporal_sill: float = 1.0,
    temporal_range: float = 1.0,
    nugget: float = 0.0,
    model: str = "exponential",
) -> dict:
    """
    Construct a separable spatio-temporal covariance matrix.

    .. math::

        C(h, u) = C_s(h) \\cdot C_t(u)

    where :math:`C_s` and :math:`C_t` are marginal spatial and temporal
    covariance functions.

    :param coords: Spatial coordinates (n, 2).
    :param times: Time points (T,).
    :param spatial_sill: Spatial sill parameter.
    :param spatial_range: Spatial range parameter.
    :param temporal_sill: Temporal sill parameter.
    :param temporal_range: Temporal range parameter.
    :param nugget: Nugget variance.
    :param model: Covariance model: ``'exponential'`` or ``'gaussian'``.
    :return: dict with ``cov_matrix`` (nT x nT), ``spatial_cov``, ``temporal_cov``.
    :raises ValueError: If model is unknown.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 9.
    """
    coords = np.asarray(coords, dtype=float)
    times = np.asarray(times, dtype=float)
    n = len(coords)
    T = len(times)

    sdists = cdist(coords, coords)
    tdists = np.abs(times[:, None] - times[None, :])

    if model == "exponential":
        Cs = spatial_sill * np.exp(-sdists / (spatial_range + 1e-12))
        Ct = temporal_sill * np.exp(-tdists / (temporal_range + 1e-12))
    elif model == "gaussian":
        Cs = spatial_sill * np.exp(-sdists ** 2 / (spatial_range ** 2 + 1e-12))
        Ct = temporal_sill * np.exp(-tdists ** 2 / (temporal_range ** 2 + 1e-12))
    else:
        raise ValueError(f"Unknown model: {model}")

    cov = np.kron(Ct, Cs)
    np.fill_diagonal(cov, cov.diagonal() + nugget)

    return {
        "cov_matrix": cov,
        "spatial_cov": Cs,
        "temporal_cov": Ct,
        "n": n,
        "T": T,
        "model": model,
    }


stsep_fn = stsep


def cheatsheet() -> str:
    return "stsep({}) -> Separable spatio-temporal covariance matrix."
