# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""
Conditional Average Treatment Effect (CATE) via T-learner or S-learner.

Implements ``estimate_cate`` — estimates per-unit heterogeneous treatment
effects using meta-learner strategies with Random Forest base learners.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler


def estimate_cate(
    data: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    covariates: list[str],
    meta_learner: str = "t_learner",
) -> pd.Series:
    r"""Estimate per-unit Conditional Average Treatment Effects (CATE).

    The CATE for unit *i* is :math:`\\tau(X_i) = \\mathbb{E}[Y(1) - Y(0) \\mid X = X_i]`.

    **T-learner** (default): fits separate outcome models on treated and
    control units, then computes :math:`\\hat{\\tau}(X_i) = \\hat{\\mu}_1(X_i)
    - \\hat{\\mu}_0(X_i)`.

    **S-learner**: fits a single outcome model on all units with treatment
    as a feature, then computes
    :math:`\\hat{\\tau}(X_i) = \\hat{\\mu}(X_i, T=1) - \\hat{\\mu}(X_i, T=0)`.

    :param data: Input DataFrame.
    :type data: pandas.DataFrame
    :param treatment: Binary treatment column (0/1).
    :type treatment: str
    :param outcome: Outcome column.
    :type outcome: str
    :param covariates: Covariate column names.
    :type covariates: list[str]
    :param meta_learner: ``"t_learner"`` or ``"s_learner"`` (default ``"t_learner"``).
    :type meta_learner: str
    :return: Series of per-unit CATE estimates, indexed like the input.
    :rtype: pandas.Series

    References
    ----------
    Kunzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019).
    Metalearners for estimating heterogeneous treatment effects using
    machine learning. *PNAS*, 116(10), 4156--4165.
    https://doi.org/10.1073/pnas.1804597116
    """
    frame = data[[treatment, outcome, *covariates]].dropna().copy()
    t = frame[treatment].values.astype(float)
    y = frame[outcome].values.astype(float)

    # Preprocess covariates
    X_raw = frame[covariates].copy()
    for col in X_raw.columns:
        if not pd.api.types.is_numeric_dtype(X_raw[col]):
            le = LabelEncoder()
            X_raw[col] = le.fit_transform(X_raw[col].astype(str))
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    treated_mask = t == 1
    control_mask = t == 0

    if meta_learner == "t_learner":
        # T-learner: separate models per arm
        rf1 = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf0 = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf1.fit(X[treated_mask], y[treated_mask])
        rf0.fit(X[control_mask], y[control_mask])
        mu1 = rf1.predict(X)
        mu0 = rf0.predict(X)

    elif meta_learner == "s_learner":
        # S-learner: single model with treatment as feature
        X_with_t = np.column_stack([X, t])
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X_with_t, y)
        X_treat = np.column_stack([X, np.ones(len(X))])
        X_ctrl = np.column_stack([X, np.zeros(len(X))])
        mu1 = rf.predict(X_treat)
        mu0 = rf.predict(X_ctrl)

    else:
        raise ValueError(f"Unknown meta_learner: {meta_learner!r}. Choose 't_learner' or 's_learner'.")

    cate = mu1 - mu0
    return pd.Series(cate, index=frame.index, name="cate")


cate = estimate_cate


def cheatsheet() -> str:
    return "estimate_cate({}) -> Conditional Average Treatment Effect (CATE) via T-learner or"
