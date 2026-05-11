# morie.fn — function file (hadesllm/morie)
"""Response surface methodology (2nd-order polynomial fit)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def response_surface(
    X: np.ndarray,
    y: np.ndarray,
    degree: int = 2,
) -> DescriptiveResult:
    """Fit a response surface model via ordinary least squares.

    Constructs a full second-order polynomial model with linear,
    interaction, and quadratic terms:

    .. math::

        y = \\beta_0 + \\sum_i \\beta_i x_i
        + \\sum_{i \\leq j} \\beta_{ij} x_i x_j + \\varepsilon

    Parameters
    ----------
    X : ndarray
        Design matrix (n x p), *p* factors.
    y : ndarray
        Response vector (n,).
    degree : int, default 2
        Polynomial degree (1 or 2).

    Returns
    -------
    DescriptiveResult
        ``value`` is the R-squared.  ``extra`` has ``coefficients``,
        ``predicted``, ``residuals``, ``adj_r_squared``, ``n_terms``.

    Raises
    ------
    ValueError
        If X and y dimensions are incompatible or degree not in {1,2}.

    References
    ----------
    Box, G. E. P., & Wilson, K. B. (1951). On the experimental
    attainment of optimum conditions. *Journal of the Royal
    Statistical Society B*, 13(1), 1--45.

    Myers, R. H., Montgomery, D. C., & Anderson-Cook, C. M. (2016).
    *Response Surface Methodology* (4th ed.). Wiley.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] != len(y):
        raise ValueError("X rows must match y length.")
    if degree not in (1, 2):
        raise ValueError(f"degree must be 1 or 2, got {degree}.")

    n, p = X.shape
    cols = [np.ones(n)]

    for j in range(p):
        cols.append(X[:, j])

    if degree == 2:
        for i in range(p):
            for j in range(i, p):
                cols.append(X[:, i] * X[:, j])

    Z = np.column_stack(cols)
    beta, residuals_arr, rank, sv = np.linalg.lstsq(Z, y, rcond=None)

    y_hat = Z @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n_terms = Z.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_terms) if n > n_terms else 0.0

    return DescriptiveResult(
        name="ResponseSurface",
        value=float(r2),
        extra={
            "coefficients": beta,
            "predicted": y_hat,
            "residuals": y - y_hat,
            "adj_r_squared": float(adj_r2),
            "n_terms": n_terms,
            "degree": degree,
        },
    )


rsm = response_surface


def cheatsheet() -> str:
    return "response_surface({}) -> Response surface methodology (2nd-order polynomial fit)."
