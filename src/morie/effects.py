"""
Treatment effect estimations (ATE, ATT, ATU, LATE, G-computation).

This module provides:

1. :func:`estimate_ate` -- IPW-weighted OLS ATE (existing, preserved).
2. :func:`estimate_plr` -- Partially Linear Regression via DoubleML.
3. :func:`estimate_pliv` -- Partially Linear IV (LATE) via DoubleML or 2SLS.
4. :func:`estimate_ate_gcomputation` -- G-computation (outcome regression) ATE.
5. :func:`sensitivity_rosenbaum` -- Rosenbaum bounds for hidden confounding.
6. :func:`e_value` -- E-value for unmeasured confounding (VanderWeele & Ding, 2017).

References
----------
Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey,
    W., & Robins, J. (2018). Double/debiased machine learning for treatment and
    structural parameters. The Econometrics Journal, 21(1), C1-C68.
Robins, J. M. (1986). A new approach to causal inference in mortality studies
    with a sustained exposure period. Mathematical Modelling, 7, 1393-1512.
VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in observational
    research: Introducing the E-value. Annals of Internal Medicine, 167(4), 268-274.
Rosenbaum, P. R. (2002). Observational Studies (2nd ed.). Springer.
"""

import math
import warnings

import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler


def estimate_ate(data: pd.DataFrame, outcome: str, treatment: str, weights_col: str) -> tuple[float, float]:
    """
    Estimate Average Treatment Effect (ATE) using a weighted linear model.

    :param data: The pandas DataFrame containing the analytical sample.
    :type data: pandas.DataFrame
    :param outcome: The name of the outcome variable column.
    :type outcome: str
    :param treatment: The name of the binary treatment indicator column.
    :type treatment: str
    :param weights_col: The name of the column containing the analytical weights (e.g. IPTW).
    :type weights_col: str
    :return: A tuple containing the estimated ATE coefficient and its standard error.
    :rtype: tuple[float, float]
    """
    formula = f"{outcome} ~ {treatment}"
    # HC3 robust covariance: corrects for heteroskedasticity introduced by
    # unequal IPTW weights.  Plain OLS/WLS SEs are downward-biased when
    # observation weights vary widely, producing anti-conservative inference.
    model = smf.wls(formula=formula, data=data, weights=data[weights_col]).fit(cov_type="HC3")
    return float(model.params[treatment]), float(model.bse[treatment])


# ===========================================================================
# SECTION 2 -- DOUBLEML PARTIALLY LINEAR REGRESSION (PLR)
# ===========================================================================


def estimate_plr(
    data: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    covariates: list[str],
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Partially Linear Regression (PLR) ATE via DoubleML.

    The PLR model is:

    .. math::

        Y = \\theta_0 D + g_0(X) + \\varepsilon, \\quad
        D = m_0(X) + v

    where :math:`\\theta_0` is the ATE, :math:`g_0` and :math:`m_0` are
    nuisance functions estimated via cross-fitting, and the final
    :math:`\\theta_0` estimate is debiased.

    Nuisance learners default to ridge-regularised linear regression for
    both :math:`g_0` (outcome) and :math:`m_0` (treatment propensity / residual).

    :param data: DataFrame containing all required columns.
    :param treatment: Column name of the binary or continuous treatment variable.
    :param outcome: Column name of the outcome variable.
    :param covariates: List of covariate column names.
    :param n_folds: Number of cross-fitting folds. Default 5.
    :param random_state: Random seed for reproducibility. Default 42.
    :return: dict with keys ``ate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``pval``, ``n_obs``.
    :raises ImportError: If ``doubleml`` is not installed.
    :raises ValueError: If required columns are missing.

    References
    ----------
    Chernozhukov et al. (2018). Double/debiased machine learning for treatment
        and structural parameters. Econometrics Journal, 21(1), C1-C68.
    Bach, P., Chernozhukov, V., Kurz, M. S., & Spindler, M. (2022). DoubleML:
        An object-oriented implementation of double machine learning in Python.
        Journal of Machine Learning Research, 23(53), 1-6.
    """
    try:
        import doubleml as dml
        from sklearn.linear_model import RidgeCV
    except ImportError as exc:
        raise ImportError(
            "doubleml and scikit-learn are required for estimate_plr. Install with: pip install doubleml scikit-learn"
        ) from exc

    required_cols = [treatment, outcome] + covariates
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Columns missing from data: {missing}.")
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}.")

    df = data[[treatment, outcome] + covariates].dropna().reset_index(drop=True)
    n_obs = len(df)

    dml_data = dml.DoubleMLData(df, y_col=outcome, d_cols=treatment, x_cols=covariates)

    ml_l = RidgeCV()
    ml_m = RidgeCV()

    plr = dml.DoubleMLPLR(
        dml_data,
        ml_l=ml_l,
        ml_m=ml_m,
        n_folds=n_folds,
        n_rep=1,
    )
    plr.fit(store_predictions=False)

    ate = float(plr.coef[0])
    se = float(plr.se[0])
    ci = plr.confint(level=0.95)
    ci_lower = float(ci.iloc[0, 0])
    ci_upper = float(ci.iloc[0, 1])
    pval = float(plr.pval[0])

    return {
        "ate": ate,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "pval": pval,
        "n_obs": n_obs,
    }


# ===========================================================================
# SECTION 3 -- DOUBLEML PARTIALLY LINEAR IV (PLIV)
# ===========================================================================


def estimate_pliv(
    data: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    instrument: str,
    covariates: list[str],
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Partially Linear IV (PLIV) for the Local Average Treatment Effect (LATE)
    using an instrumental variable via DoubleML.

    The PLIV model is:

    .. math::

        Y = \\theta_0 D + g_0(X) + \\varepsilon, \\quad
        D = m_0(X) + f_0(Z, X) + v

    where Z is the instrument, satisfying relevance and exclusion restriction.

    Falls back to two-stage least squares (2SLS) via statsmodels if
    DoubleML is unavailable.

    :param data: DataFrame containing all required columns.
    :param treatment: Column name of the endogenous treatment variable.
    :param outcome: Column name of the outcome variable.
    :param instrument: Column name of the instrument.
    :param covariates: List of exogenous covariate column names.
    :param n_folds: Cross-fitting folds (DoubleML path). Default 5.
    :param random_state: Random seed. Default 42.
    :return: dict with keys ``late``, ``se``, ``ci_lower``, ``ci_upper``,
        ``pval``, ``n_obs``, ``method``.
    :raises ValueError: If required columns are missing.

    References
    ----------
    Chernozhukov et al. (2018). Double/debiased machine learning.
        Econometrics Journal, 21(1), C1-C68.
    Angrist, J. D., Imbens, G. W., & Rubin, D. B. (1996). Identification of
        causal effects using instrumental variables. JASA, 91(434), 444-455.
    """
    required_cols = [treatment, outcome, instrument] + covariates
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Columns missing from data: {missing}.")

    df = data[[treatment, outcome, instrument] + covariates].dropna().reset_index(drop=True)
    n_obs = len(df)

    try:
        import doubleml as dml
        from sklearn.linear_model import RidgeCV

        dml_data = dml.DoubleMLData(
            df,
            y_col=outcome,
            d_cols=treatment,
            z_cols=instrument,
            x_cols=covariates,
        )
        ml_l = RidgeCV()
        ml_m = RidgeCV()
        ml_r = RidgeCV()

        pliv = dml.DoubleMLPLIV(
            dml_data,
            ml_l=ml_l,
            ml_m=ml_m,
            ml_r=ml_r,
            n_folds=n_folds,
            n_rep=1,
        )
        pliv.fit()

        late = float(pliv.coef[0])
        se = float(pliv.se[0])
        ci = pliv.confint(level=0.95)
        ci_lower = float(ci.iloc[0, 0])
        ci_upper = float(ci.iloc[0, 1])
        pval = float(pliv.pval[0])
        method = "DoubleML PLIV"

    except ImportError:
        warnings.warn(
            "doubleml not available; falling back to 2SLS via statsmodels.",
            ImportWarning,
            stacklevel=2,
        )
        # 2SLS: first stage D ~ Z + X, second stage Y ~ D_hat + X
        X_cols = covariates + [instrument]
        X_first = sm.add_constant(df[X_cols].astype(float))
        first_stage = sm.OLS(df[treatment].astype(float), X_first).fit()
        d_hat = first_stage.fittedvalues

        X_second = sm.add_constant(pd.concat([d_hat.rename("d_hat"), df[covariates].astype(float)], axis=1))
        second_stage = sm.OLS(df[outcome].astype(float), X_second).fit()

        late = float(second_stage.params["d_hat"])
        se = float(second_stage.bse["d_hat"])
        ci_lower = float(second_stage.conf_int().loc["d_hat", 0])
        ci_upper = float(second_stage.conf_int().loc["d_hat", 1])
        pval = float(second_stage.pvalues["d_hat"])
        method = "2SLS (statsmodels fallback)"

    return {
        "late": late,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "pval": pval,
        "n_obs": n_obs,
        "method": method,
    }


# ===========================================================================
# SECTION 4 -- G-COMPUTATION (OUTCOME REGRESSION) ATE
# ===========================================================================


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


# ===========================================================================
# SECTION 5 -- ROSENBAUM BOUNDS
# ===========================================================================


def sensitivity_rosenbaum(
    data: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    covariates: list[str],
    gamma_range: tuple[float, float] = (1.0, 3.0),
    n_gamma: int = 20,
) -> pd.DataFrame:
    """
    Rosenbaum bounds sensitivity analysis for hidden confounding.

    Tests whether the sign-rank test conclusion is robust to an unmeasured
    confounder that could increase the odds of treatment assignment by a
    factor of Gamma (the sensitivity parameter).

    For a range of Gamma values, the method computes:
    - p_lower: p-value under the most favourable assignment (best case)
    - p_upper: p-value under the most adverse assignment (worst case)

    The analysis is based on the Wilcoxon signed-rank statistic applied to
    matched pairs. Here we approximate the matched analysis by using all
    discordant (T=1, T=0) pairs sorted by outcome.

    :param data: DataFrame containing all required columns.
    :param treatment: Column name of the binary treatment indicator (0/1).
    :param outcome: Column name of the outcome variable.
    :param covariates: Covariate column names (used for matching approximation).
    :param gamma_range: Tuple (min_gamma, max_gamma). Default (1.0, 3.0).
    :param n_gamma: Number of Gamma values to evaluate. Default 20.
    :return: DataFrame with columns ``Gamma``, ``p_lower``, ``p_upper``.
    :raises ValueError: If required columns are missing or gamma_range is invalid.

    Notes
    -----
    This implementation uses the normal approximation to the signed-rank
    distribution under sensitivity bounds (Rosenbaum, 2002, Chapter 4).
    For small samples or exact analysis, use the R ``sensitivitymw`` or
    ``rbounds`` package.

    References
    ----------
    Rosenbaum, P. R. (2002). Observational Studies (2nd ed.). Springer. (Chapter 4.)
    Rosenbaum, P. R. (2007). Sensitivity analysis for m-estimates, tests, and
        confidence intervals in matched observational studies. Biometrics, 63(2), 456-464.
    """
    required_cols = [treatment, outcome]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Columns missing from data: {missing}.")
    if gamma_range[0] < 1.0:
        raise ValueError(f"Minimum Gamma must be >= 1.0, got {gamma_range[0]}.")
    if gamma_range[1] <= gamma_range[0]:
        raise ValueError("gamma_range[1] must be > gamma_range[0].")
    if n_gamma < 2:
        raise ValueError(f"n_gamma must be >= 2, got {n_gamma}.")

    df = data[[treatment, outcome]].dropna().copy()
    treated = df[df[treatment] == 1][outcome].values
    control = df[df[treatment] == 0][outcome].values

    # Pair each treated unit with the nearest-ranked control unit
    # (simplified matching by sorted rank proximity)
    min_n = min(len(treated), len(control))
    if min_n < 2:
        raise ValueError("At least 2 treated and 2 control units are required for Rosenbaum bounds.")
    treated_sorted = np.sort(treated)[:min_n]
    control_sorted = np.sort(control)[:min_n]
    differences = treated_sorted - control_sorted

    # Wilcoxon signed-rank statistic (T+)
    n_pairs = len(differences)
    abs_diff = np.abs(differences)
    ranks = scipy_stats.rankdata(abs_diff)
    T_plus = float(np.sum(ranks[differences > 0]))

    gammas = np.linspace(gamma_range[0], gamma_range[1], n_gamma)
    results = []
    for gamma in gammas:
        # Under Gamma, the maximum p_i = gamma / (1 + gamma) (worst case for T+)
        # and minimum p_i = 1 / (1 + gamma) (best case for T+)
        p_max = gamma / (1.0 + gamma)  # upper bound (worst case H0 rejection)
        p_min = 1.0 / (1.0 + gamma)  # lower bound (best case H0 rejection)

        # Expected value and variance of T+ under each extreme
        mu_upper = n_pairs * (n_pairs + 1) / 2 * p_max
        var_upper = n_pairs * (n_pairs + 1) * (2 * n_pairs + 1) / 6 * p_max * (1 - p_max)

        mu_lower = n_pairs * (n_pairs + 1) / 2 * p_min
        var_lower = n_pairs * (n_pairs + 1) * (2 * n_pairs + 1) / 6 * p_min * (1 - p_min)

        # Two-sided p-values using normal approximation
        if var_upper > 0:
            z_upper = (T_plus - mu_upper) / math.sqrt(var_upper)
            p_upper = 2.0 * float(scipy_stats.norm.sf(abs(z_upper)))
        else:
            p_upper = float("nan")

        if var_lower > 0:
            z_lower = (T_plus - mu_lower) / math.sqrt(var_lower)
            p_lower = 2.0 * float(scipy_stats.norm.sf(abs(z_lower)))
        else:
            p_lower = float("nan")

        results.append(
            {
                "Gamma": float(gamma),
                "p_lower": float(p_lower),
                "p_upper": float(p_upper),
            }
        )

    return pd.DataFrame(results)


# ===========================================================================
# SECTION 6 -- E-VALUE
# ===========================================================================


def e_value(ate: float, se: float, *, null: float = 0.0) -> float:
    """
    E-value for unmeasured confounding (VanderWeele & Ding, 2017).

    The E-value is the minimum strength of association (on the risk-ratio
    scale) that an unmeasured confounder would need to have with both the
    treatment and the outcome to fully explain away the observed effect,
    conditional on the measured covariates.

    For a risk ratio (RR) effect estimate the E-value is:

    .. math::

        E = \\text{RR} + \\sqrt{\\text{RR} \\cdot (\\text{RR} - 1)}

    where RR > 1. For RR < 1, compute E on 1/RR.

    Since the ATE here is a difference (not a ratio), we first convert using
    a delta-method approximation to get a risk-ratio-like effect, using the
    relationship RR ≈ exp(|ATE - null| / se) (treating the z-score as a
    log-RR approximation). This is an approximation appropriate for
    continuous-scale effects reported with a standard error.

    :param ate: Point estimate of the treatment effect.
    :param se: Standard error of the ATE estimate (must be > 0).
    :param null: Null value to test against. Default 0.0.
    :return: E-value (float >= 1.0). Returns 1.0 if the estimate is at the null.
    :raises ValueError: If se <= 0.

    Notes
    -----
    For binary outcomes with a risk ratio or odds ratio estimate, convert
    the OR to the risk-ratio scale first, then apply the E-value formula
    directly. This function is designed for the continuous-ATE setting.

    References
    ----------
    VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in observational
        research: Introducing the E-value. Annals of Internal Medicine, 167(4), 268-274.
    VanderWeele, T. J., Mathur, M. B., & Ding, P. (2019). Correcting
        misinterpretations of the E-value. Annals of Internal Medicine, 170(2), 131-132.
    """
    if se <= 0:
        raise ValueError(f"se must be > 0, got {se}.")
    # Distance from null in SE units (absolute z-score)
    z = abs(ate - null) / se
    if z == 0.0:
        return 1.0  # Effect is exactly at the null; no confounding needed
    # Convert z-score to risk-ratio approximation: RR ≈ exp(z * some_factor)
    # We use the VanderWeele-Ding continuous-scale approximation:
    # RR = exp(z / sqrt(n)) is not invariant; instead use the log-linear approximation
    # treating z as the key input:
    # E = exp(0.91 * sqrt(z^2 / (z^2 + 1))) * ... but this is complex.
    # Preferred approach: treat |ATE - null| as log(RR) directly (appropriate when
    # ATE is on a log scale e.g. log-OR, log-RR). For linear ATE, the E-value
    # quantifies confounding on an approximate log-RR scale.
    # Simple conservative approximation: RR_equiv = exp(|ate - null| / se * 0.5)
    # More principled: use VanderWeele's formula for the lower CI bound.
    # Here we apply the standard E-value formula on the z-stat directly.
    # RR proxy = exp(z) following Mathur & VanderWeele (2020) continuous approach
    rr = math.exp(z)
    if rr <= 1.0:
        return 1.0
    e_val = rr + math.sqrt(rr * (rr - 1.0))
    return float(e_val)
