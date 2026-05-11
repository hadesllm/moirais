# morie.fn — function file (hadesllm/morie)
"""EM algorithm imputation for multivariate normal data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def em_impute(
    data: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    n_iter: int = 50,
    tol: float = 1e-6,
) -> pd.DataFrame:
    r"""
    Impute missing values via the Expectation-Maximisation (EM) algorithm
    assuming multivariate normality.

    **E-step**: For each row with missing entries, compute the conditional
    expectation of the missing variables given the observed variables and
    the current parameter estimates (:math:`\mu`, :math:`\Sigma`).

    **M-step**: Update :math:`\mu` and :math:`\Sigma` using the filled data
    (including the conditional expectations from the E-step) and the
    conditional covariance corrections (Dempster, Laird, & Rubin, 1977).

    Convergence is declared when the maximum absolute change in :math:`\mu`
    between iterations falls below *tol*, or after *n_iter* iterations.

    Only numeric columns are imputed; non-numeric columns are returned
    unchanged.

    :param data: Input DataFrame (may contain NaN in numeric columns).
    :type data: pandas.DataFrame
    :param columns: Columns to impute.  Default: all numeric columns with
        missing values.
    :type columns: list[str] or None
    :param n_iter: Maximum number of EM iterations.  Default 50.
    :type n_iter: int
    :param tol: Convergence tolerance on the mean vector.  Default 1e-6.
    :type tol: float
    :return: Completed DataFrame with NaN values filled by EM estimates.
    :rtype: pandas.DataFrame
    :raises ValueError: If fewer than 2 numeric columns or no missing values.

    References
    ----------
    Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood
    from incomplete data via the EM algorithm. *Journal of the Royal
    Statistical Society: Series B*, 39(1), 1--38.
    https://doi.org/10.1111/j.2517-6161.1977.tb01600.x

    Little, R. J. A., & Rubin, D. B. (2019). *Statistical Analysis with
    Missing Data* (3rd ed.). Wiley. https://doi.org/10.1002/9781119482260
    """
    result = data.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()

    if columns is not None:
        use_cols = [c for c in columns if c in numeric_cols]
    else:
        use_cols = numeric_cols

    if len(use_cols) < 2:
        raise ValueError("Need at least 2 numeric columns for EM imputation.")

    mat = result[use_cols].values.astype(float)
    n, p = mat.shape

    has_missing = np.isnan(mat).any()
    if not has_missing:
        raise ValueError("No missing values found in the specified columns.")

    # Initialise: fill NaN with column means
    col_means = np.nanmean(mat, axis=0)
    for j in range(p):
        mask_j = np.isnan(mat[:, j])
        mat[mask_j, j] = col_means[j]

    mu = np.mean(mat, axis=0)
    sigma = np.cov(mat, rowvar=False, ddof=1)
    # Regularise
    sigma += np.eye(p) * 1e-10

    original_nan = np.isnan(data[use_cols].values.astype(float))

    for _iteration in range(n_iter):
        mu_old = mu.copy()
        mat_filled = mat.copy()
        cov_correction = np.zeros((p, p))

        for i in range(n):
            miss_idx = np.where(original_nan[i])[0]
            obs_idx = np.where(~original_nan[i])[0]

            if len(miss_idx) == 0:
                continue
            if len(obs_idx) == 0:
                # All missing: fill with current mean
                mat_filled[i, miss_idx] = mu[miss_idx]
                cov_correction[np.ix_(miss_idx, miss_idx)] += sigma[np.ix_(miss_idx, miss_idx)]
                continue

            # Conditional distribution: miss | obs
            sigma_mm = sigma[np.ix_(miss_idx, miss_idx)]
            sigma_mo = sigma[np.ix_(miss_idx, obs_idx)]
            sigma_oo = sigma[np.ix_(obs_idx, obs_idx)]

            try:
                sigma_oo_inv = np.linalg.inv(sigma_oo + np.eye(len(obs_idx)) * 1e-10)
            except np.linalg.LinAlgError:
                sigma_oo_inv = np.linalg.pinv(sigma_oo)

            # Conditional mean
            x_obs = data[use_cols].values[i, obs_idx].astype(float)
            cond_mean = mu[miss_idx] + sigma_mo @ sigma_oo_inv @ (x_obs - mu[obs_idx])
            mat_filled[i, miss_idx] = cond_mean

            # Conditional covariance (for M-step correction)
            cond_cov = sigma_mm - sigma_mo @ sigma_oo_inv @ sigma_mo.T
            cov_correction[np.ix_(miss_idx, miss_idx)] += cond_cov

        # M-step
        mu = np.mean(mat_filled, axis=0)
        sigma = np.cov(mat_filled, rowvar=False, ddof=1)
        sigma += cov_correction / n
        sigma += np.eye(p) * 1e-10

        mat = mat_filled

        if np.max(np.abs(mu - mu_old)) < tol:
            break

    result[use_cols] = mat
    return result


em_i = em_impute


def cheatsheet() -> str:
    return "em_impute({}) -> EM algorithm imputation for multivariate normal data."
