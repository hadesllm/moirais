# morie.fn — function file (hadesllm/morie)
"""Penalized spline regression."""

from __future__ import annotations

import numpy as np


def pensp(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_knots: int | None = None,
    penalty: float | None = None,
    degree: int = 3,
    x_new: np.ndarray | None = None,
    seed: int = 42,
) -> dict:
    r"""
    Penalized spline (P-spline) regression estimator.

    Fits a spline basis expansion with a roughness penalty that
    balances fidelity to the data against smoothness:

    .. math::

        \hat{\beta} = \arg\min_{\beta}
        \| y - B \beta \|^2 + \lambda \, \beta^T D^T D \beta

    where :math:`B` is a truncated-power or B-spline basis matrix,
    :math:`D` is a second-order difference matrix on the coefficients,
    and :math:`\lambda` is the smoothing parameter.

    When ``penalty=None``, the smoothing parameter is selected by
    leave-one-out cross-validation (LOO-CV).

    :param x: Predictor array, shape ``(n,)``.
    :param y: Response array, shape ``(n,)``.
    :param n_knots: Number of interior knots. Default ``min(35, n // 4)``.
    :param penalty: Smoothing parameter :math:`\lambda`. If ``None``,
        selected by LOO-CV.
    :param degree: Polynomial degree of the spline. Default 3 (cubic).
    :param x_new: Prediction grid. Default uses sorted *x*.
    :param seed: Random seed (reserved). Default 42.
    :return: dict with ``x_grid``, ``y_hat``, ``coefficients``,
        ``penalty``, ``n_knots``, ``gcv_score``.
    :raises ValueError: If input dimensions are wrong.

    References
    ----------
    Eilers, P. H. C. & Marx, B. D. (1996). Flexible smoothing with
        B-splines and penalties. *Statistical Science*, 11(2), 89--121.
    Ruppert, D., Wand, M. P. & Carroll, R. J. (2003).
        *Semiparametric Regression*. Cambridge University Press.
    Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods
        in Econometrics*. Springer, Section 2.4.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-D arrays.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length.")
    n = x.shape[0]

    if n_knots is None:
        n_knots = min(35, n // 4)
    n_knots = max(1, n_knots)

    knots = np.linspace(x.min(), x.max(), n_knots + 2)[1:-1]

    def _basis(xv):
        cols = [np.ones_like(xv)]
        for d in range(1, degree + 1):
            cols.append(xv ** d)
        for kn in knots:
            cols.append(np.maximum(xv - kn, 0) ** degree)
        return np.column_stack(cols)

    B = _basis(x)
    p = B.shape[1]

    D = np.diff(np.eye(p), n=2, axis=0)
    DtD = D.T @ D

    def _fit(lam):
        A = B.T @ B + lam * DtD
        beta = np.linalg.solve(A, B.T @ y)
        return beta

    def _gcv(lam):
        A = B.T @ B + lam * DtD
        H = B @ np.linalg.solve(A, B.T)
        y_hat = H @ y
        resid = y - y_hat
        tr_H = np.trace(H)
        denom = (1 - tr_H / n) ** 2
        if denom <= 0:
            return np.inf
        return float(np.mean(resid ** 2) / denom)

    if penalty is None:
        candidates = np.logspace(-4, 6, 50)
        gcv_scores = [_gcv(lam) for lam in candidates]
        best_idx = int(np.argmin(gcv_scores))
        penalty = float(candidates[best_idx])
        gcv_score = gcv_scores[best_idx]
    else:
        gcv_score = _gcv(penalty)

    beta = _fit(penalty)
    if x_new is None:
        x_new = np.sort(x)
    else:
        x_new = np.asarray(x_new, dtype=float)

    B_new = _basis(x_new)
    y_hat = B_new @ beta

    return {
        "x_grid": x_new,
        "y_hat": y_hat,
        "coefficients": beta,
        "penalty": penalty,
        "n_knots": n_knots,
        "gcv_score": gcv_score,
    }


pensp_fn = pensp


def cheatsheet() -> str:
    return "pensp(x, y) -> Penalized spline regression."
