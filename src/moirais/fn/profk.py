# moirais.fn — function file (hadesllm/moirais)
"""Profile likelihood for semiparametric models.

Computes the profile likelihood by maximizing over nuisance parameters
for each fixed value of the parameter of interest. Used for confidence
intervals and inference in semiparametric settings.

References
----------
Murphy, S. A. & van der Vaart, A. W. (2000). On profile likelihood.
*Journal of the American Statistical Association*, 95(450), 449--465.

Kosorok, M. R. (2008). *Introduction to Empirical Processes and
Semiparametric Inference*. Springer. Chapters 13--14.

Severini, T. A. & Wong, W. H. (1992). Profile likelihood and
conditionally parametric models. *Annals of Statistics*, 20(4),
1768--1802.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def profk(
    Y: np.ndarray,
    X: np.ndarray,
    *,
    theta_grid: np.ndarray | None = None,
    n_grid: int = 50,
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""Compute the profile likelihood for the leading regression coefficient.

    For the partially linear model :math:`Y = \theta X_1 + g(X_{-1}) + \varepsilon`,
    the profile likelihood fixes :math:`\theta` and profiles over the
    nonparametric component :math:`g`:

    .. math::

        \ell_p(\theta) = \max_{g} \ell(\theta, g)
            = -\frac{n}{2} \log \hat{\sigma}^2(\theta)

    where :math:`\hat{\sigma}^2(\theta) = n^{-1}\|Y - \theta X_1 - \hat{g}_\theta\|^2`.

    The profile-likelihood-based CI is :math:`\{\theta : 2[\ell_p(\hat\theta) - \ell_p(\theta)] \le \chi^2_1(1-\alpha)\}`.

    Parameters
    ----------
    Y : np.ndarray
        Outcome vector, shape ``(n,)``.
    X : np.ndarray
        Covariate matrix, shape ``(n, p)`` where the first column is the
        parameter of interest.
    theta_grid : np.ndarray | None
        Grid of theta values to evaluate.  If *None*, auto-generated
        around the OLS estimate.
    n_grid : int
        Number of grid points if *theta_grid* is not supplied.
    alpha : float
        Significance level for the profile likelihood CI.

    Returns
    -------
    dict[str, Any]
        ``theta_hat``, ``profile_ll`` (array), ``theta_grid`` (array),
        ``ci_lower``, ``ci_upper``, ``n``, ``method``.
    """
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape

    beta_ols = np.linalg.lstsq(X, Y, rcond=None)[0]
    theta_hat = float(beta_ols[0])
    se_ols = float(np.sqrt(np.sum((Y - X @ beta_ols)**2) / max(n - p, 1) *
                           np.linalg.pinv(X.T @ X)[0, 0]))

    if theta_grid is None:
        margin = max(4.0 * se_ols, 0.5)
        theta_grid = np.linspace(theta_hat - margin, theta_hat + margin, n_grid)

    profile_ll = np.empty(len(theta_grid))
    X1 = X[:, 0]
    X_rest = X[:, 1:] if p > 1 else np.ones((n, 1))
    if p == 1:
        X_rest = np.ones((n, 1))

    for i, theta in enumerate(theta_grid):
        resid = Y - theta * X1
        if X_rest.shape[1] > 0:
            beta_nuisance = np.linalg.lstsq(X_rest, resid, rcond=None)[0]
            fitted_resid = resid - X_rest @ beta_nuisance
        else:
            fitted_resid = resid - np.mean(resid)
        sigma2 = float(np.mean(fitted_resid**2))
        sigma2 = max(sigma2, 1e-300)
        profile_ll[i] = -0.5 * n * np.log(sigma2)

    max_ll = float(np.max(profile_ll))
    deviance = 2.0 * (max_ll - profile_ll)
    chi2_crit = stats.chi2.ppf(1.0 - alpha, df=1)

    in_ci = theta_grid[deviance <= chi2_crit]
    ci_lower = float(in_ci[0]) if len(in_ci) > 0 else float(theta_grid[0])
    ci_upper = float(in_ci[-1]) if len(in_ci) > 0 else float(theta_grid[-1])

    return {
        "theta_hat": theta_hat,
        "profile_ll": profile_ll,
        "theta_grid": theta_grid,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n": n,
        "method": "ProfileLikelihood",
    }


profk_fn = profk


def cheatsheet() -> str:
    return "profk(Y, X) -> Profile likelihood CI (Kosorok 2008, Ch. 13-14)."
