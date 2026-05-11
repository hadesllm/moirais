"""Nonlinear utility via polynomial."""

import numpy as np

from ._containers import SpatialResult


def svnut(ideal, pos, *, coeffs=None):
    """Nonlinear utility via polynomial.

    Returns
    -------
    SpatialResult
    """

    ideal = np.asarray(ideal, dtype=float)
    pos = np.asarray(pos, dtype=float)
    d = float(np.linalg.norm(ideal - pos))
    coeffs = np.asarray(coeffs, dtype=float)
    stat = float(np.polyval(-np.abs(coeffs), d))
    return SpatialResult(
        name="Nonlinear Polynomial Utility",
        statistic=float(stat),
        extra={"distance": d},
    )


svnut = svnut  # alias


def cheatsheet() -> str:
    return "svnut({}) -> Nonlinear utility via polynomial."
