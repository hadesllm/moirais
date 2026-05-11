# morie.fn — function file (hadesllm/morie)
"""
Propensity score estimation via logistic regression.

Implements ``compute_propensity_scores`` — the foundational propensity score
model used by IPW, AIPW, ATT, ATC, and other causal estimators in morie.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler


def compute_propensity_scores(data: pd.DataFrame, treatment: str, covariates: list) -> pd.Series:
    """
    Compute propensity scores using logistic regression with standard preprocessing.

    Non-numeric (object/categorical) columns are integer-encoded with
    :class:`~sklearn.preprocessing.LabelEncoder` and all numeric columns are
    standardised with :class:`~sklearn.preprocessing.StandardScaler` before
    fitting.  Standardisation stabilises the logistic-regression solver and
    makes coefficients comparable across covariates on different scales.
    Encoding is applied in-place on a copy; the original ``data`` frame is
    not modified.

    :param data: The input pandas DataFrame containing covariates and treatment.
    :type data: pandas.DataFrame
    :param treatment: The name of the column containing the binary treatment variable.
    :type treatment: str
    :param covariates: A list of column names for the covariates to adjust for.
    :type covariates: list[str]
    :return: A pandas Series containing the predicted propensity scores
        (values in the open interval (0, 1)).
    :rtype: pandas.Series

    References
    ----------
    Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the
    propensity score in observational studies for causal effects.
    *Biometrika*, 70(1), 41--55. https://doi.org/10.1093/biomet/70.1.41
    """
    X_raw = data[covariates].copy()
    y = data[treatment]

    # Encode any non-numeric columns so LogisticRegression receives a numeric
    # matrix.  LabelEncoder maps each unique string/category to an integer.
    for col in X_raw.columns:
        if not pd.api.types.is_numeric_dtype(X_raw[col]):
            le = LabelEncoder()
            X_raw[col] = le.fit_transform(X_raw[col].astype(str))

    # Standardise all numeric columns: mean 0, unit variance.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    return pd.Series(model.predict_proba(X_scaled)[:, 1], index=data.index, name="ps")


ps_fit = compute_propensity_scores


def cheatsheet() -> str:
    return "compute_propensity_scores({}) -> Propensity score estimation via logistic regression."
