# moirais.fn — function file (hadesllm/moirais)
"""Nonparametric model selection via cross-validation."""

from __future__ import annotations

import numpy as np


def npsel(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bandwidths: np.ndarray | list[float] | None = None,
    kernel: str = "gaussian",
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    r"""
    Nonparametric bandwidth / model selection via K-fold cross-validation.

    For each candidate bandwidth :math:`h`, fits a Nadaraya-Watson
    kernel regression estimator

    .. math::

        \hat{m}_h(x) = \frac{\sum_{j} K_h(x - X_j) \, Y_j}
        {\sum_{j} K_h(x - X_j)}

    and evaluates the mean squared prediction error on held-out folds.
    The bandwidth minimising CV error is selected.

    When ``bandwidths=None``, a grid of 30 values is constructed from
    Silverman's rule-of-thumb scaled by factors in ``[0.1, 3.0]``.

    :param x: Predictor array, shape ``(n,)``.
    :param y: Response array, shape ``(n,)``.
    :param bandwidths: Candidate bandwidths. Default auto-grid.
    :param kernel: ``"gaussian"`` or ``"epanechnikov"``. Default ``"gaussian"``.
    :param n_folds: Number of CV folds. Default 5.
    :param seed: Random seed for fold assignment. Default 42.
    :return: dict with ``best_bandwidth``, ``cv_scores`` (array),
        ``bandwidths`` (array), ``best_cv_score``.
    :raises ValueError: If inputs have wrong shape or *kernel* unknown.

    References
    ----------
    Hardle, W., Hall, P. & Marron, J. S. (1988). How far are
        automatically chosen regression smoothing parameters from their
        optimum? *JASA*, 83(401), 86--95.
    Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods
        in Econometrics*. Springer, Section 2.6.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-D.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length.")
    n = x.shape[0]
    valid_kernels = {"gaussian", "epanechnikov"}
    if kernel not in valid_kernels:
        raise ValueError(f"kernel must be one of {valid_kernels}.")
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}.")

    if kernel == "gaussian":
        def K(u):
            return np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)
    else:
        def K(u):
            return np.where(np.abs(u) <= 1, 0.75 * (1 - u ** 2), 0.0)

    h_rot = 1.06 * np.std(x, ddof=1) * n ** (-0.2)
    h_rot = max(h_rot, 1e-6)

    if bandwidths is None:
        bandwidths = h_rot * np.logspace(np.log10(0.1), np.log10(3.0), 30)
    else:
        bandwidths = np.asarray(bandwidths, dtype=float)

    rng = np.random.default_rng(seed)
    folds = rng.integers(0, n_folds, size=n)

    cv_scores = np.empty(len(bandwidths))

    for hi, h in enumerate(bandwidths):
        mse_folds = []
        for fold in range(n_folds):
            test_mask = folds == fold
            train_mask = ~test_mask
            x_tr, y_tr = x[train_mask], y[train_mask]
            x_te, y_te = x[test_mask], y[test_mask]
            if x_te.shape[0] == 0 or x_tr.shape[0] == 0:
                continue
            u = (x_te[:, None] - x_tr[None, :]) / h
            w = K(u)
            w_sum = w.sum(axis=1)
            w_sum = np.where(w_sum > 0, w_sum, 1.0)
            y_pred = (w * y_tr[None, :]).sum(axis=1) / w_sum
            mse_folds.append(np.mean((y_te - y_pred) ** 2))
        cv_scores[hi] = np.mean(mse_folds) if mse_folds else np.inf

    best_idx = int(np.argmin(cv_scores))

    return {
        "best_bandwidth": float(bandwidths[best_idx]),
        "cv_scores": cv_scores,
        "bandwidths": bandwidths,
        "best_cv_score": float(cv_scores[best_idx]),
    }


npsel_fn = npsel


def cheatsheet() -> str:
    return "npsel(x, y) -> Nonparametric model selection (CV bandwidth)."
