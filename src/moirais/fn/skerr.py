"""Spatial error model (SEM) via feasible GLS."""

import numpy as np

from ._containers import DescriptiveResult


def spatial_error_model(
    y: np.ndarray,
    X: np.ndarray,
    W: np.ndarray,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> DescriptiveResult:
    """
    Estimate a spatial error model (SEM) via two-step feasible GLS.

    The model is:

    .. math::

        y = X \\beta + u, \\qquad u = \\lambda W u + \\varepsilon

    where :math:`\\lambda` is the spatial autoregressive parameter for the
    error term, :math:`W` is the spatial weights matrix, and
    :math:`\\varepsilon \\sim N(0, \\sigma^2 I)`.

    The estimation proceeds by:

    1. OLS to obtain residuals :math:`\\hat{u}`.
    2. Estimate :math:`\\lambda` by regressing :math:`\\hat{u}` on
       :math:`W\\hat{u}` (Cochrane-Orcutt style).
    3. Transform the model using :math:`(I - \\hat{\\lambda} W)` and re-estimate
       :math:`\\beta` via OLS on the transformed system.
    4. Iterate until convergence.

    :param y: Response vector, shape (n,).
    :param X: Design matrix, shape (n, p). Should include an intercept column
        if desired.
    :param W: Spatial weights matrix, shape (n, n). Typically row-standardised.
    :param max_iter: Maximum number of iterations. Default 50.
    :param tol: Convergence tolerance on :math:`\\lambda`. Default 1e-6.
    :return: :class:`DescriptiveResult` with extra containing ``coefs``,
        ``lambda_``, ``residuals``, ``sigma2``, ``iterations``.
    :raises ValueError: If shapes are incompatible.

    References
    ----------
    Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer.

    Kelejian, H. H. & Prucha, I. R. (1998). A generalized spatial two-stage
    least squares procedure for estimating a spatial autoregressive model with
    autoregressive disturbances. *J. Real Estate Finance and Economics*, 17(1),
    99-121. https://doi.org/10.1023/A:1007707430416
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)
    n = len(y)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] != n:
        raise ValueError(f"X rows ({X.shape[0]}) != y length ({n}).")
    if W.shape != (n, n):
        raise ValueError(f"W must be ({n}, {n}), got {W.shape}.")

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    lam = 0.0

    for it in range(max_iter):
        Wu = W @ resid
        denom = float(Wu @ Wu)
        lam_new = float(Wu @ resid) / denom if denom > 0 else 0.0

        I_lW = np.eye(n) - lam_new * W
        y_t = I_lW @ y
        X_t = I_lW @ X
        beta = np.linalg.lstsq(X_t, y_t, rcond=None)[0]
        resid = y - X @ beta

        if abs(lam_new - lam) < tol:
            lam = lam_new
            break
        lam = lam_new

    sigma2 = float(resid @ resid) / max(n - X.shape[1], 1)

    return DescriptiveResult(
        name="Spatial Error Model",
        value=lam,
        extra={
            "coefs": beta,
            "lambda_": lam,
            "residuals": resid,
            "sigma2": sigma2,
            "iterations": it + 1,
            "n": n,
            "p": X.shape[1],
        },
    )


skerr_fn = spatial_error_model


def cheatsheet() -> str:
    return "spatial_error_model({}) -> Spatial error model (SEM) via feasible GLS."
