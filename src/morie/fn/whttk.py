"""Whittaker smoother. 'Character is destiny. -- Heraclitus' -- General usage"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ._containers import DescriptiveResult


def whittaker_smooth(
    y: np.ndarray,
    lambda_: float = 100.0,
    d: int = 2,
) -> DescriptiveResult:
    r"""
    Whittaker smoother (penalised least squares).

    Minimises :math:`\\|y - z\\|^2 + \\lambda \\|D^d z\\|^2`
    where *D* is the *d*-th order difference matrix.

    :param y: 1-D signal array.
    :param lambda_: Smoothing penalty (larger = smoother). Default 100.
    :param d: Order of the difference penalty. Default 2.
    :return: DescriptiveResult with smoothed array in extra.
    :raises ValueError: If y is too short or lambda_ <= 0.

    References
    ----------
    Eilers, P. H. C. (2003). A perfect smoother. Analytical Chemistry,
    75(14), 3631--3636. doi:10.1021/ac034173t
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size < d + 1:
        raise ValueError(f"y must be 1-D with length >= {d + 1}.")
    if lambda_ <= 0.0:
        raise ValueError(f"lambda_ must be > 0, got {lambda_}.")

    n = len(y)
    I = sparse.eye(n, format="csc")
    D = sparse.eye(n, format="csc")
    for _ in range(d):
        D = D[1:, :] - D[:-1, :]

    W = I + lambda_ * D.T @ D
    z = spsolve(W, y)

    residual = y - z
    rmse = float(np.sqrt(np.mean(residual**2)))

    return DescriptiveResult(
        name="Whittaker Smoother",
        value=rmse,
        extra={
            "smoothed": z,
            "residuals": residual,
            "rmse": rmse,
            "lambda": lambda_,
            "difference_order": d,
            "n": n,
        },
    )


whttk = whittaker_smooth


def cheatsheet() -> str:
    return "whittaker_smooth({}) -> Whittaker smoother. 'Character is destiny. -- Heraclitus' -- General "
