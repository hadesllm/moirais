# morie.fn -- function file (hadesllm/morie)
"""Parametric g-formula (g-computation)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import ESRes


def g_formula(
    df: pd.DataFrame,
    *,
    outcome: str = "Y",
    treatment: str = "A",
    covariates: list[str] | None = None,
) -> ESRes:
    """
    Parametric g-formula (g-computation) for the ATE.

    Fits a logistic outcome model, then predicts counterfactual
    outcomes under treatment and control for all subjects.

    Parameters
    ----------
    df : DataFrame
        Data with outcome, treatment, and covariates.
    outcome : str
        Binary outcome column.
    treatment : str
        Binary treatment column.
    covariates : list of str, optional
        Covariate columns.

    Returns
    -------
    ESRes
        estimate = ATE.

    References
    ----------
    Robins, J. M. (1986). A new approach to causal inference in
    mortality studies with a sustained exposure period. *Math
    Modelling*, 7(9-12), 1393-1512.
    """
    if outcome not in df.columns or treatment not in df.columns:
        raise ValueError("outcome and treatment must be in DataFrame.")

    if covariates is None:
        covariates = [c for c in df.select_dtypes(include=[np.number]).columns if c not in (outcome, treatment)]

    Y = df[outcome].values.astype(float)
    A = df[treatment].values.astype(float)
    X = df[covariates].values.astype(float) if covariates else np.ones((len(Y), 1))

    XA = np.column_stack([np.ones(len(Y)), A, X])
    from numpy.linalg import lstsq

    beta, _, _, _ = lstsq(XA, Y, rcond=None)

    X1 = np.column_stack([np.ones(len(Y)), np.ones(len(Y)), X])
    X0 = np.column_stack([np.ones(len(Y)), np.zeros(len(Y)), X])

    Y1 = X1 @ beta
    Y0 = X0 @ beta

    ate = float(np.mean(Y1 - Y0))
    se = float(np.std(Y1 - Y0, ddof=1) / np.sqrt(len(Y)))

    return ESRes(
        measure="g_formula_ATE",
        estimate=ate,
        ci_lower=ate - 1.96 * se,
        ci_upper=ate + 1.96 * se,
        se=se,
        n=len(Y),
    )


gform = g_formula


def cheatsheet() -> str:
    return "g_formula({}) -> Parametric g-formula (g-computation)."
