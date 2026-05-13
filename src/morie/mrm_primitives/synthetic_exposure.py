"""Synthetic small-area-estimated exposure offset.

Adapted from Laniyonu & Goff (2021) BMC Psychiatry 21(1):500.  The
key trick: when you need a rate-per-hidden-subpopulation (force-per-
PwSMI, contact-per-undocumented, contact-per-homeless) and no
administrative census of that subpopulation exists, you can:

  1. Fit P(trait | covariates) on a national probability sample
     (NCS-R for SMI; ACS-style survey for other traits) using ONLY
     covariates also available at the area level.
  2. Apply the fitted coefficients to area-level marginals from the
     ACS / census to predict P(trait) per area.
  3. Multiply by area-level adult population to get a synthetic
     "population at risk" denominator.

Generalises far beyond Laniyonu & Goff's SMI application:

  - homelessness rates of police force
  - LGBTQ stop-and-frisk rates
  - undocumented-immigrant ICE-contact rates
  - any "rate per hidden subpopulation" estimand

The primitive returns the per-area exposure offset suitable for use
in a Poisson / negative-binomial GLM as the ``exposure=`` argument.

We intentionally use a simple logistic via :mod:`scipy.optimize` so
the primitive runs on the base morie install (no statsmodels
dependency-of-dependency).  Callers who want fancier survey-design
weighting can pass a fitted estimator via the ``fit_callable``
argument.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def synthetic_area_exposure(
    *,
    survey_df: pd.DataFrame,
    survey_trait_col: str,
    survey_covariate_cols: list[str],
    area_df: pd.DataFrame,
    area_population_col: str,
    fit_callable: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    return_per_area_rate: bool = False,
) -> pd.Series | tuple[pd.Series, pd.Series]:
    """Compute synthetic exposure offset per area.

    Parameters
    ----------
    survey_df : pd.DataFrame
        Survey microdata (e.g., NCS-R long-form).  One row per
        respondent.  Must contain ``survey_trait_col`` (0/1) and the
        named covariates.
    survey_trait_col : str
        Binary column indicating the trait of interest (e.g., SMI).
    survey_covariate_cols : list[str]
        Covariates available in BOTH the survey and the area dataset.
        Logistic coefficients are fit on the survey and applied to
        the area marginals.
    area_df : pd.DataFrame
        One row per area (tract, precinct, etc.).  Must contain the
        same covariate columns as ``survey_covariate_cols`` (as
        area-level *means* / *proportions*).
    area_population_col : str
        Adult-population column in ``area_df``.
    fit_callable : callable, optional
        Override for the default logistic fit.  Signature:
        ``fit_callable(X, y) -> coef``, where ``coef`` is a vector of
        length len(survey_covariate_cols) + 1 (intercept first).
    return_per_area_rate : bool, default False
        If True, returns a tuple (exposure_offset, predicted_rate).

    Returns
    -------
    pd.Series indexed by ``area_df.index``: synthetic exposure offset
        = predicted P(trait | area marginals) × area population.
        Suitable for ``exposure=`` in a Poisson / negative-binomial
        GLM that counts trait-specific events.

    Notes
    -----
    This is "synthetic" because the trait is not directly observed at
    the area level — it's predicted from a national probability
    sample.  Cite the underlying survey explicitly when reporting.
    """
    # 1. Fit P(trait | covariates) on the survey
    if fit_callable is None:
        coef = _logistic_fit(
            survey_df[survey_covariate_cols].to_numpy(dtype=float),
            survey_df[survey_trait_col].to_numpy(dtype=float),
        )
    else:
        coef = fit_callable(
            survey_df[survey_covariate_cols].to_numpy(dtype=float),
            survey_df[survey_trait_col].to_numpy(dtype=float),
        )

    # 2. Apply to area marginals
    X_area = area_df[survey_covariate_cols].to_numpy(dtype=float)
    X_area_with_int = np.column_stack([np.ones(X_area.shape[0]), X_area])
    eta = X_area_with_int @ coef
    pred_rate = 1.0 / (1.0 + np.exp(-eta))

    # 3. Multiply by population to get exposure offset
    exposure = pred_rate * area_df[area_population_col].to_numpy(dtype=float)
    exposure_series = pd.Series(exposure, index=area_df.index,
                                name="synthetic_exposure")

    if return_per_area_rate:
        rate_series = pd.Series(pred_rate, index=area_df.index,
                                name="predicted_rate")
        return exposure_series, rate_series
    return exposure_series


def _logistic_fit(X: np.ndarray, y: np.ndarray, max_iter: int = 200,
                  tol: float = 1e-6) -> np.ndarray:
    """Minimal-dependency logistic regression via Newton-IRLS.

    Returns coef of length p+1 (intercept first).  Adequate for the
    SAE use case where the survey is well-conditioned.  Falls back
    gracefully on non-convergence by returning the last iterate.
    """
    n, p = X.shape
    X_int = np.column_stack([np.ones(n), X])
    beta = np.zeros(p + 1)
    for _ in range(max_iter):
        eta = X_int @ beta
        # Stable sigmoid
        mu = np.where(eta >= 0,
                      1.0 / (1.0 + np.exp(-eta)),
                      np.exp(eta) / (1.0 + np.exp(eta)))
        # Weights for IRLS
        w = mu * (1 - mu)
        # Guard against zero-variance weight rows
        w = np.clip(w, 1e-10, None)
        XW = X_int * w[:, None]
        XWX = XW.T @ X_int
        XWz = XW.T @ (X_int @ beta + (y - mu) / w)
        try:
            new_beta = np.linalg.solve(XWX, XWz)
        except np.linalg.LinAlgError:
            return beta  # singular Hessian — return last good iterate
        if np.max(np.abs(new_beta - beta)) < tol:
            return new_beta
        beta = new_beta
    return beta
