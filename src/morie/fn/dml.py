# morie.fn -- function file (hadesllm/morie)
"""
Double Machine Learning (DML) -- Partially Linear Regression Model.

Implements ``estimate_double_ml`` -- estimates the ATE using DoubleML's PLR
with Random Forest nuisance estimators and cross-fitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

_DOUBLEML_RANDOM_STATE: int = 42
"""Module-level seed for all DoubleML estimations.  Change at call-site if needed."""

_DOUBLEML_N_FOLDS: int = 5
"""Number of cross-fitting folds.  Must be >= 2."""

_DOUBLEML_N_REP: int = 1
"""Number of repeated cross-fitting repetitions.  Set > 1 to reduce variance."""


def estimate_double_ml(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    covariates: list,
    *,
    random_state: int = _DOUBLEML_RANDOM_STATE,
    n_folds: int = _DOUBLEML_N_FOLDS,
    n_rep: int = _DOUBLEML_N_REP,
):
    """
    Estimate the Average Treatment Effect using Double Machine Learning (DML).

    Uses :class:`doubleml.DoubleMLPLR` (Partially Linear Regression Model)
    with Random Forest nuisance estimators for both the outcome regression
    (``ml_l``) and the treatment model (``ml_m``).

    Reproducibility
    ---------------
    All stochastic operations are seeded deterministically:

    1. ``numpy`` global seed is set to ``random_state`` immediately before
       constructing the learners.
    2. Both ``RandomForestRegressor`` instances receive ``random_state``.
    3. The ``DoubleMLPLR`` object is constructed with ``n_folds`` and
       ``n_rep`` passed explicitly so that the cross-fitting schedule is
       fixed for a given seed.

    To change the seed for a sensitivity run, pass ``random_state=<int>``.

    :param data: The dataset containing outcome, treatment, and covariates.
    :type data: pandas.DataFrame
    :param outcome: Name of the continuous outcome variable.
    :type outcome: str
    :param treatment: Name of the treatment variable.
    :type treatment: str
    :param covariates: List of covariate column names to control for.
    :type covariates: list[str]
    :param random_state: Integer seed for all RNGs.  Default 42.
    :type random_state: int, optional
    :param n_folds: Number of cross-fitting folds.  Default 5.
    :type n_folds: int, optional
    :param n_rep: Number of repeated cross-fitting repetitions.  Default 1.
    :type n_rep: int, optional
    :return: A fitted :class:`doubleml.DoubleMLPLR` object containing the
        causal estimate and inference results.
    :rtype: doubleml.DoubleMLPLR

    References
    ----------
    Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
    Newey, W., & Robins, J. (2018). Double/debiased machine learning for
    treatment and structural parameters. *The Econometrics Journal*, 21(1),
    C1--C68. https://doi.org/10.1111/ectj.12097
    """
    try:
        import doubleml as dml
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "DoubleML is required for estimate_double_ml(). "
            "Install the project dependencies in the repo venv before using this function."
        ) from exc

    # Seed numpy's global RNG before constructing the learners so that the
    # internal random state of the DoubleML cross-fitting splits is fully
    # deterministic for a given random_state value.
    np.random.seed(random_state)

    dml_data = dml.DoubleMLData(data, y_col=outcome, d_cols=treatment, x_cols=covariates)

    # ml_l: outcome regression E[Y|X] -- always a regressor.
    # ml_m: treatment model P(T=1|X) -- classifier for binary treatment so that
    #       DoubleML receives probability predictions, not continuous predictions.
    #       Using a Regressor here would silently fit E[T|X] as a number in [0,1]
    #       rather than a probability, producing incorrect Neyman-orthogonal scores.
    ml_l = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=random_state)
    ml_m = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_state)

    dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=n_folds, n_rep=n_rep)
    dml_plr.fit()

    return dml_plr


dml = estimate_double_ml


def cheatsheet() -> str:
    return "estimate_double_ml({}) -> Double Machine Learning (DML) -- Partially Linear Regression "
