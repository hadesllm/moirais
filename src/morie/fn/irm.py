# morie.fn — function file (hadesllm/morie)
"""
Interactive Regression Model (IRM) via DoubleML.

Implements ``estimate_irm`` — estimates the ATE allowing for treatment effect
heterogeneity using Neyman-orthogonal scores and cross-fitting.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def estimate_irm(
    data: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    covariates: list[str],
    n_folds: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    r"""Estimate the ATE via the Interactive Regression Model (IRM) using DoubleML.

    The IRM extends the partially linear model by allowing treatment effect
    heterogeneity.  It models:

    .. math::

        Y = g_0(T, X) + U, \\quad \\mathbb{E}[U \\mid X, T] = 0

        T = m_0(X) + V, \\quad \\mathbb{E}[V \\mid X] = 0

    where :math:`g_0` is the outcome regression and :math:`m_0` is the
    propensity score.  The Neyman-orthogonal score for the ATE is:

    .. math::

        \\psi = g_0(1, X) - g_0(0, X)
        + \\frac{T(Y - g_0(1,X))}{m_0(X)}
        - \\frac{(1-T)(Y - g_0(0,X))}{1 - m_0(X)} - \\theta

    :param data: Input DataFrame.
    :type data: pandas.DataFrame
    :param treatment: Binary treatment column.
    :type treatment: str
    :param outcome: Outcome column.
    :type outcome: str
    :param covariates: Covariate column names.
    :type covariates: list[str]
    :param n_folds: Number of cross-fitting folds (default 5).
    :type n_folds: int
    :param random_state: Random seed (default 42).
    :type random_state: int
    :return: Dictionary with ``ate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``n``, ``method``.
    :rtype: dict[str, Any]

    References
    ----------
    Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
    Newey, W., & Robins, J. (2018). Double/debiased machine learning for
    treatment and structural parameters. *The Econometrics Journal*, 21(1),
    C1--C68. https://doi.org/10.1111/ectj.12097
    """
    try:
        import doubleml as dml
    except ImportError as exc:
        raise ImportError("DoubleML is required for estimate_irm(). Install with: pip install DoubleML") from exc

    frame = data[[treatment, outcome, *covariates]].dropna().copy()

    # Encode non-numeric covariates
    for col in covariates:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            le = LabelEncoder()
            frame[col] = le.fit_transform(frame[col].astype(str))

    np.random.seed(random_state)

    dml_data = dml.DoubleMLData(
        frame,
        y_col=outcome,
        d_cols=treatment,
        x_cols=covariates,
    )

    # ml_g: outcome regression E[Y|T,X] -- regressor
    # ml_m: propensity score P(T=1|X) -- classifier
    ml_g = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=random_state)
    ml_m = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_state)

    dml_irm = dml.DoubleMLIRM(dml_data, ml_g, ml_m, n_folds=n_folds)
    dml_irm.fit()

    ate = float(dml_irm.coef[0])
    se = float(dml_irm.se[0])
    z = 1.959964

    return {
        "ate": ate,
        "se": se,
        "ci_lower": ate - z * se,
        "ci_upper": ate + z * se,
        "n": len(frame),
        "method": "IRM (DoubleML)",
    }


irm = estimate_irm


def cheatsheet() -> str:
    return "estimate_irm({}) -> Interactive Regression Model (IRM) via DoubleML."
