# morie.fn — function file (hadesllm/morie)
"""G-computation (outcome regression) ATE estimator with bootstrap SE."""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler


def estimate_ate_gcomputation(
    data: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    covariates: list[str],
    outcome_model: str = "linear",
) -> dict:
    """
    G-computation (outcome regression / standardisation) ATE estimator.

    The G-computation estimator proceeds in three steps:

    1. Fit an outcome model :math:`E[Y | T, X]` on the observed data.
    2. Predict potential outcomes :math:`\\hat{Y}(1)` and :math:`\\hat{Y}(0)`
       for every unit by setting T = 1 and T = 0 respectively.
    3. Compute ATE as the average difference:

    .. math::

        \\widehat{\\text{ATE}} = \\frac{1}{n} \\sum_i
        \\left(\\hat{Y}_i(1) - \\hat{Y}_i(0)\\right)

    Standard error is estimated via non-parametric bootstrap (500 iterations
    with seed = 42) on the full three-step procedure.

    :param data: DataFrame containing all required columns.
    :param treatment: Column name of the binary treatment indicator (0/1).
    :param outcome: Column name of the outcome variable.
    :param covariates: List of covariate column names.
    :param outcome_model: ``"linear"`` (OLS) for continuous outcomes or
        ``"logistic"`` for binary outcomes. Default ``"linear"``.
    :return: dict with keys ``ate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``n_obs``, ``outcome_model``.
    :raises ValueError: If required columns are missing or outcome_model is invalid.

    References
    ----------
    Robins, J. M. (1986). A new approach to causal inference in mortality
        studies. Mathematical Modelling, 7, 1393-1512.
    Hernan, M. A., & Robins, J. M. (2020). Causal Inference: What If.
        Chapman & Hall/CRC. (Chapter 13.)
    """
    valid_models = {"linear", "logistic"}
    if outcome_model not in valid_models:
        raise ValueError(f"outcome_model must be one of {valid_models}.")

    required_cols = [treatment, outcome] + covariates
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Columns missing from data: {missing}.")

    df = data[[treatment, outcome] + covariates].dropna().reset_index(drop=True)
    n_obs = len(df)
    if n_obs < 10:
        raise ValueError("G-computation requires at least 10 complete observations.")

    feature_cols = [treatment] + covariates

    def _fit_and_predict_ate(df_boot: pd.DataFrame) -> float:
        X = df_boot[feature_cols].astype(float).values
        y = df_boot[outcome].astype(float).values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if outcome_model == "linear":
            model = LinearRegression()
        else:
            model = LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)

        model.fit(X_scaled, y)

        # Counterfactual datasets: all treated / all control
        X_t1 = df_boot[feature_cols].astype(float).copy()
        X_t0 = df_boot[feature_cols].astype(float).copy()
        X_t1[treatment] = 1.0
        X_t0[treatment] = 0.0

        X_t1_scaled = scaler.transform(X_t1.values)
        X_t0_scaled = scaler.transform(X_t0.values)

        if outcome_model == "linear":
            y1_hat = model.predict(X_t1_scaled)
            y0_hat = model.predict(X_t0_scaled)
        else:
            y1_hat = model.predict_proba(X_t1_scaled)[:, 1]
            y0_hat = model.predict_proba(X_t0_scaled)[:, 1]

        return float(np.mean(y1_hat - y0_hat))

    # Point estimate
    ate = _fit_and_predict_ate(df)

    # Bootstrap SE (500 iterations, seeded for reproducibility)
    rng = np.random.default_rng(42)
    boot_ates = []
    for _ in range(500):
        idx = rng.integers(0, n_obs, size=n_obs)
        boot_df = df.iloc[idx].reset_index(drop=True)
        try:
            boot_ates.append(_fit_and_predict_ate(boot_df))
        except Exception:
            continue

    if len(boot_ates) < 50:
        warnings.warn(
            "Fewer than 50 successful bootstrap iterations; SE may be unreliable.",
            stacklevel=2,
        )

    se = float(np.std(boot_ates, ddof=1)) if len(boot_ates) > 1 else float("nan")
    ci_lower = float(np.percentile(boot_ates, 2.5)) if boot_ates else float("nan")
    ci_upper = float(np.percentile(boot_ates, 97.5)) if boot_ates else float("nan")

    return {
        "ate": ate,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_obs": n_obs,
        "outcome_model": outcome_model,
    }


g_comp_fn = estimate_ate_gcomputation


def cheatsheet() -> str:
    return "estimate_ate_gcomputation({}) -> G-computation (outcome regression) ATE estimator with bootst"
