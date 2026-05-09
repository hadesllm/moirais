"""Sieve estimation via series/polynomial approximation."""

from __future__ import annotations

import numpy as np


def sieve(
    x: np.ndarray,
    y: np.ndarray,
    *,
    basis: str = "polynomial",
    k: int | None = None,
    x_new: np.ndarray | None = None,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    r"""
    Sieve estimator of a nonparametric regression function.

    Approximates the unknown regression function :math:`g(x)` by
    projecting onto a finite-dimensional sieve space that grows with
    the sample size.

    .. math::

        \hat{g}_K(x) = \sum_{j=0}^{K} \hat{\beta}_j \, p_j(x)

    where :math:`p_j` are basis functions (polynomial, cosine/Fourier,
    or B-spline) and :math:`K = K_n \to \infty` as :math:`n \to \infty`
    at an appropriate rate.

    The default number of basis terms is :math:`K = \lfloor n^{1/5} \rfloor`
    following Chen (2007) rate-optimal guidance.

    :param x: Predictor array, shape ``(n,)``.
    :param y: Response array, shape ``(n,)``.
    :param basis: ``"polynomial"``, ``"cosine"``, or ``"bspline"``.
        Default ``"polynomial"``.
    :param k: Number of basis terms. Default ``int(n ** 0.2)``.
    :param x_new: Points at which to predict. Default uses sorted *x*.
    :param alpha: Significance level for pointwise CIs. Default 0.05.
    :param seed: Random seed (unused, reserved for future). Default 42.
    :return: dict with ``x_grid``, ``y_hat``, ``coefficients``, ``k``,
        ``basis``, ``residual_var``.
    :raises ValueError: If *basis* is unknown or lengths mismatch.

    References
    ----------
    Chen, X. (2007). Large sample sieve estimation of semi-nonparametric
        models. In *Handbook of Econometrics*, Vol. 6B, Ch. 76.
    Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods
        in Econometrics*. Springer, Ch. 2.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-D arrays.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length.")
    n = x.shape[0]
    valid_bases = {"polynomial", "cosine", "bspline"}
    if basis not in valid_bases:
        raise ValueError(f"basis must be one of {valid_bases}, got '{basis}'.")

    if k is None:
        k = max(1, int(n ** 0.2))

    def _build_basis(xv: np.ndarray, k_: int) -> np.ndarray:
        if basis == "polynomial":
            return np.column_stack([xv ** j for j in range(k_ + 1)])
        elif basis == "cosine":
            cols = [np.ones_like(xv)]
            x_scaled = (xv - xv.min()) / max(xv.max() - xv.min(), 1e-12)
            for j in range(1, k_ + 1):
                cols.append(np.cos(np.pi * j * x_scaled))
            return np.column_stack(cols)
        else:
            from scipy.interpolate import BSpline

            t_interior = np.linspace(xv.min(), xv.max(), k_ - 2) if k_ > 2 else np.array([])
            degree = min(3, k_)
            knots = np.concatenate([
                np.repeat(xv.min(), degree + 1),
                t_interior,
                np.repeat(xv.max(), degree + 1),
            ])
            n_basis = len(knots) - degree - 1
            cols = []
            for j in range(n_basis):
                c = np.zeros(n_basis)
                c[j] = 1.0
                spl = BSpline(knots, c, degree, extrapolate=True)
                cols.append(spl(xv))
            return np.column_stack(cols) if cols else np.ones((len(xv), 1))

    B = _build_basis(x, k)
    beta, res, _, _ = np.linalg.lstsq(B, y, rcond=None)
    y_fitted = B @ beta
    resid = y - y_fitted
    sigma2 = float(np.sum(resid ** 2) / max(n - B.shape[1], 1))

    if x_new is None:
        x_new = np.sort(x)
    else:
        x_new = np.asarray(x_new, dtype=float)

    B_new = _build_basis(x_new, k)
    y_hat = B_new @ beta

    return {
        "x_grid": x_new,
        "y_hat": y_hat,
        "coefficients": beta,
        "k": k,
        "basis": basis,
        "residual_var": sigma2,
    }


sieve_fn = sieve


def cheatsheet() -> str:
    return "sieve(x, y) -> Sieve estimation (series/polynomial approximation)."
