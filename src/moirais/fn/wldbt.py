"""Wild bootstrap for heteroskedastic regression."""

from __future__ import annotations

import numpy as np


def wldbt(
    y: np.ndarray,
    X: np.ndarray,
    *,
    n_boot: int = 999,
    alpha: float = 0.05,
    distribution: str = "rademacher",
    coef_index: int = 1,
    seed: int = 42,
) -> dict:
    r"""
    Wild bootstrap test and CI for a regression coefficient under
    heteroskedasticity of unknown form.

    Fits :math:`y = X\beta + \varepsilon` by OLS, computes residuals
    :math:`\hat{e}_i`, and generates bootstrap pseudo-samples

    .. math::

        y_i^* = X_i \hat{\beta} + \hat{e}_i \, v_i

    where :math:`v_i` are i.i.d. draws from a two-point distribution
    (Rademacher or Mammen) satisfying :math:`E[v]=0`, :math:`E[v^2]=1`.
    The bootstrap distribution of :math:`\hat{\beta}_{j}^*` is used for
    inference on the *j*-th coefficient (``coef_index``).

    :param y: Response vector, shape ``(n,)``.
    :param X: Design matrix, shape ``(n, p)``. Include a column of ones
        for the intercept.
    :param n_boot: Number of bootstrap replicates. Default 999.
    :param alpha: Significance level. Default 0.05.
    :param distribution: ``"rademacher"`` (:math:`\pm 1` with equal
        probability) or ``"mammen"`` (two-point distribution matching
        the first three moments of :math:`N(0,1)`). Default ``"rademacher"``.
    :param coef_index: Index of the coefficient to test (0-based).
        Default 1 (first non-intercept).
    :param seed: Random seed. Default 42.
    :return: dict with ``estimate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``p_value``, ``n_boot``.
    :raises ValueError: If dimensions are inconsistent or *distribution*
        is unknown.

    References
    ----------
    Wu, C. F. J. (1986). Jackknife, bootstrap and other resampling
        methods in regression analysis. *Annals of Statistics*, 14(4),
        1261--1295.
    Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods
        in Econometrics*. Springer, Section 3.5.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if y.ndim != 1 or X.ndim != 2:
        raise ValueError("y must be 1-D, X must be 2-D.")
    n, p = X.shape
    if n != y.shape[0]:
        raise ValueError("y and X must have the same number of rows.")
    if coef_index < 0 or coef_index >= p:
        raise ValueError(f"coef_index {coef_index} out of range [0, {p}).")
    valid_dist = {"rademacher", "mammen"}
    if distribution not in valid_dist:
        raise ValueError(f"distribution must be one of {valid_dist}.")

    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta_hat
    resid = y - y_hat

    rng = np.random.default_rng(seed)
    boot_coefs = np.empty(n_boot)

    for b in range(n_boot):
        if distribution == "rademacher":
            v = rng.choice([-1.0, 1.0], size=n)
        else:
            s5 = np.sqrt(5)
            p_neg = (s5 + 1) / (2 * s5)
            vals = [-(s5 - 1) / 2, (s5 + 1) / 2]
            v = rng.choice(vals, size=n, p=[p_neg, 1 - p_neg])

        y_star = y_hat + resid * v
        beta_star = np.linalg.lstsq(X, y_star, rcond=None)[0]
        boot_coefs[b] = beta_star[coef_index]

    se = float(np.std(boot_coefs, ddof=1))
    lo = alpha / 2
    hi = 1 - alpha / 2
    ci_lo = float(np.percentile(boot_coefs, lo * 100))
    ci_hi = float(np.percentile(boot_coefs, hi * 100))

    t_obs = beta_hat[coef_index] / se if se > 0 else 0.0
    boot_t = (boot_coefs - np.mean(boot_coefs)) / se if se > 0 else np.zeros(n_boot)
    p_value = float(np.mean(np.abs(boot_t) >= np.abs(t_obs)))

    return {
        "estimate": float(beta_hat[coef_index]),
        "se": se,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "p_value": p_value,
        "n_boot": n_boot,
    }


wldbt_fn = wldbt


def cheatsheet() -> str:
    return "wldbt(y, X) -> Wild bootstrap for heteroskedastic regression."
