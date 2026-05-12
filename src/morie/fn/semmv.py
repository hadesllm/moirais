# morie.fn -- function file (hadesllm/morie)
"""Spatial error model via maximum likelihood."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def spatial_error_ml(X: np.ndarray, y: np.ndarray, W: np.ndarray, max_iter: int = 50, tol: float = 1e-6, cdf=None) -> SpatialResult:
    r"""Spatial error model estimated by concentrated maximum likelihood.

    Fits:

    .. math::

        y = X\beta + u, \quad u = \lambda W u + \varepsilon,
        \quad \varepsilon \sim N(0, \sigma^2 I)

    Equivalently:

    .. math::

        (I - \lambda W) y = (I - \lambda W) X \beta + \varepsilon

    Estimation uses concentrated log-likelihood over :math:`\lambda`:

    .. math::

        \ell_c(\lambda) = -\frac{n}{2}\ln(2\pi)
        - \frac{n}{2}\ln\hat{\sigma}^2(\lambda)
        + \ln|I - \lambda W|

    The log-determinant is computed from eigenvalues of *W*.

    Parameters
    ----------
    X : np.ndarray
        Design matrix, shape ``(n, p)``. Include intercept if desired.
    y : np.ndarray
        Response, shape ``(n,)``.
    W : np.ndarray
        Row-standardized spatial weights matrix, shape ``(n, n)``.
    max_iter : int
        Maximum iterations for golden-section search. Default 50.
    tol : float
        Convergence tolerance for lambda. Default 1e-6.

    Returns
    -------
    SpatialResult
        ``statistic`` is the estimated :math:`\lambda`.
        ``p_value`` is the asymptotic p-value for :math:`H_0: \lambda=0`.
        ``extra`` contains ``beta``, ``sigma2``, ``log_likelihood``,
        ``residuals``, ``aic``.

    References
    ----------
    Anselin L (1988). *Spatial Econometrics: Methods and Models*.
    Kluwer Academic Publishers.

    Ord JK (1975). Estimation methods for models of spatial interaction.
    *Journal of the American Statistical Association*, 70(349), 120-126.

    LeSage JP, Pace RK (2009). *Introduction to Spatial Econometrics*.
    CRC Press.
    """
    Xm = np.asarray(X, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64).ravel()
    Wm = np.asarray(W, dtype=np.float64)
    n = len(yv)

    if Xm.ndim == 1:
        Xm = Xm.reshape(-1, 1)
    if Xm.shape[0] != n or Wm.shape != (n, n):
        raise ValueError("Dimension mismatch")

    p = Xm.shape[1]

    eig_w = np.real(np.linalg.eigvals(Wm))
    lam_lo = 1.0 / min(eig_w.min(), -1e-10) + 1e-6
    lam_hi = 1.0 / max(eig_w.max(), 1e-10) - 1e-6
    lam_lo = max(lam_lo, -0.99)
    lam_hi = min(lam_hi, 0.99)

    I = np.eye(n)

    def _concentrated_ll(lam):
        A = I - lam * Wm
        y_star = A @ yv
        X_star = A @ Xm
        beta, res, _, _ = np.linalg.lstsq(X_star, y_star, rcond=None)
        e = y_star - X_star @ beta
        sigma2 = np.dot(e, e) / n
        if sigma2 <= 0:
            return -np.inf, beta, sigma2
        log_det = np.sum(np.log(np.abs(1.0 - lam * eig_w)))
        ll = -0.5 * n * np.log(2 * np.pi) - 0.5 * n * np.log(sigma2) + log_det - 0.5 * n
        return ll, beta, sigma2

    gr = (np.sqrt(5) + 1) / 2
    a, b = lam_lo, lam_hi
    for _ in range(max_iter):
        if b - a < tol:
            break
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        llc, _, _ = _concentrated_ll(c)
        lld, _, _ = _concentrated_ll(d)
        if llc > lld:
            b = d
        else:
            a = c

    lam_hat = (a + b) / 2.0
    ll_hat, beta_hat, sigma2_hat = _concentrated_ll(lam_hat)

    ll_0, _, _ = _concentrated_ll(0.0)
    lr_stat = 2.0 * (ll_hat - ll_0)
    from scipy.stats import chi2 as _chi2

    p_value = float(1.0 - _chi2.cdf(max(lr_stat, 0), df=1))

    A_hat = I - lam_hat * Wm
    residuals = A_hat @ yv - A_hat @ Xm @ beta_hat
    aic = -2 * ll_hat + 2 * (p + 2)

    return SpatialResult(
        name="Spatial_Error_ML",
        statistic=float(lam_hat),
        p_value=p_value,
        extra={
            "beta": beta_hat,
            "sigma2": float(sigma2_hat),
            "log_likelihood": float(ll_hat),
            "residuals": residuals,
            "aic": float(aic),
            "lambda": float(lam_hat),
        },
    )


def cheatsheet() -> str:
    return "spatial_error_ml({}) -> Spatial error model via maximum likelihood."
