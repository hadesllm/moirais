# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""
Augmented Inverse Probability Weighting (AIPW) doubly-robust estimator.

Implements ``estimate_aipw`` — estimates the ATE with double robustness:
consistent if either the propensity score model or outcome model is correct.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

from morie.fn.ps_fit import compute_propensity_scores


def estimate_aipw(
    data: pd.DataFrame,
    *,
    treatment: str = "cannabis_any_use",
    outcome: str = "heavy_drinking_30d",
    covariates: list[str] | None = None,
    outcome_model: str = "logistic",
) -> dict[str, Any]:
    """
    Estimate the ATE via the Augmented Inverse Probability Weighting (AIPW)
    doubly-robust estimator.

    The influence-function score for unit *i* is:

    .. math::

        \\psi_i =
            \\hat{\\mu}_1(X_i) - \\hat{\\mu}_0(X_i)
            + \\frac{T_i \\bigl(Y_i - \\hat{\\mu}_1(X_i)\\bigr)}{\\hat{e}_i}
            - \\frac{(1-T_i)\\bigl(Y_i - \\hat{\\mu}_0(X_i)\\bigr)}{1 - \\hat{e}_i}

    The ATE estimator is :math:`\\widehat{\\text{ATE}}_\\text{AIPW} = n^{-1} \\sum_i \\psi_i`
    and the standard error is :math:`\\hat{\\sigma}_{\\psi} / \\sqrt{n}`.

    **Double robustness**: the estimator is consistent if *either* the
    propensity score model :math:`\\hat{e}(X)` or the outcome model
    :math:`\\hat{\\mu}(T, X)` is correctly specified -- not necessarily both.

    :param data: The input DataFrame.
    :type data: pandas.DataFrame
    :param treatment: Binary treatment column (0/1).
    :type treatment: str
    :param outcome: Outcome column.
    :type outcome: str
    :param covariates: Covariate column names.  Defaults to the standard
        CPADS confounders.
    :type covariates: list[str] | None
    :param outcome_model: ``"logistic"`` for binary outcomes, ``"linear"``
        for continuous.  Defaults to ``"logistic"``.
    :type outcome_model: str
    :return: Dictionary with keys ``ate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``n``, ``method``.
    :rtype: dict[str, Any]

    References
    ----------
    Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of
    regression coefficients when some regressors are not always observed.
    *Journal of the American Statistical Association*, 89(427), 846--866.

    Scharfstein, D. O., Rotnitzky, A., & Robins, J. M. (1999). Adjusting for
    nonignorable drop-out using semiparametric nonresponse models. *JASA*,
    94(448), 1096--1120.
    """
    covariates = covariates or [
        "age_group",
        "gender",
        "province_region",
        "mental_health",
        "physical_health",
    ]
    required = [treatment, outcome, *covariates]
    frame = data.loc[:, required].dropna().copy()

    t = frame[treatment].values.astype(float)
    y = frame[outcome].values.astype(float)

    # -- Propensity scores -------------------------------------------------------
    ps = compute_propensity_scores(frame, treatment=treatment, covariates=covariates).values
    ps = ps.clip(0.01, 0.99)

    # -- Outcome model: preprocess covariates the same way as propensity ---------
    X_raw = frame[covariates].copy()
    for col in X_raw.columns:
        if not pd.api.types.is_numeric_dtype(X_raw[col]):
            le = LabelEncoder()
            X_raw[col] = le.fit_transform(X_raw[col].astype(str))
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    treated_mask = t == 1
    control_mask = t == 0

    if outcome_model == "logistic":
        om1 = LogisticRegression(max_iter=1000)
        om0 = LogisticRegression(max_iter=1000)
        om1.fit(X[treated_mask], y[treated_mask])
        om0.fit(X[control_mask], y[control_mask])
        mu1 = om1.predict_proba(X)[:, 1]
        mu0 = om0.predict_proba(X)[:, 1]
    else:
        om1 = LinearRegression()
        om0 = LinearRegression()
        om1.fit(X[treated_mask], y[treated_mask])
        om0.fit(X[control_mask], y[control_mask])
        mu1 = om1.predict(X)
        mu0 = om0.predict(X)

    # -- AIPW influence scores ---------------------------------------------------
    psi = mu1 - mu0 + t * (y - mu1) / ps - (1.0 - t) * (y - mu0) / (1.0 - ps)

    n = len(psi)
    ate = float(psi.mean())
    se = float(psi.std(ddof=1) / np.sqrt(n))
    z = 1.959964  # 97.5th percentile of standard normal

    return {
        "ate": ate,
        "se": se,
        "ci_lower": ate - z * se,
        "ci_upper": ate + z * se,
        "n": n,
        "method": "AIPW (doubly robust)",
    }


aipw = estimate_aipw


def cheatsheet() -> str:
    return "estimate_aipw({}) -> Augmented Inverse Probability Weighting (AIPW) doubly-robust"
