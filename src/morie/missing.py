"""
Missing data analysis, diagnostics, and imputation for epidemiological datasets.

This module provides the complete analytical pipeline for handling missing data
in observational health studies:

1. **Pattern analysis** -- identify monotone, non-monotone, and systematic
   missingness; classify mechanisms (MCAR, MAR, MNAR).
2. **Diagnostics** -- Little's MCAR test, missingness pattern matrices,
   proportion summaries.
3. **Single imputation** -- mean, median, mode, LOCF, NOCB, regression, hot
   deck, cold deck.
4. **Multiple imputation** -- MICE (multivariate imputation by chained
   equations) with PMM, Bayesian linear regression, logistic regression for
   binary outcomes, polytomous regression for categorical outcomes, and random
   forest imputation.
5. **Pooling** -- Rubin's rules for combining multiply imputed estimates.
6. **Convergence** -- trace-plot data, R-hat diagnostics.
7. **Sensitivity analysis** -- tipping-point analysis, delta adjustment.

References
----------
Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*. Wiley.
van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.).
    CRC Press.
Little, R. J. A. (1988). A test of missing completely at random for
    multivariate data with missing values. *JASA*, 83(404), 1198--1202.
White, I. R., Royston, P., & Wood, A. M. (2011). Multiple imputation using
    chained equations: Issues and guidance for practice. *Statistics in
    Medicine*, 30(4), 377--399.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.linear_model import BayesianRidge, LinearRegression, LogisticRegression

logger = logging.getLogger(__name__)


# ===================================================================
# Result containers
# ===================================================================


@dataclass
class MissingProfile:
    """Summary of missingness in a dataset.

    Parameters
    ----------
    n_obs : int
        Total number of observations.
    n_vars : int
        Total number of variables.
    total_cells : int
        n_obs * n_vars.
    n_missing : int
        Total missing cells.
    pct_missing : float
        Percentage of missing cells.
    complete_cases : int
        Number of rows with zero missing values.
    per_variable : DataFrame
        Per-variable missingness summary.
    pattern_matrix : ndarray
        Binary matrix (1 = observed, 0 = missing).
    is_monotone : bool
        Whether the missingness pattern is monotone.
    """

    n_obs: int
    n_vars: int
    total_cells: int
    n_missing: int
    pct_missing: float
    complete_cases: int
    per_variable: pd.DataFrame
    pattern_matrix: np.ndarray
    is_monotone: bool


@dataclass
class MCARTestResult:
    """Result of Little's MCAR test."""

    test_statistic: float
    p_value: float
    df: int
    n_patterns: int
    method: str = "Little's MCAR test"


@dataclass
class ImputedDataset:
    """A single imputed dataset with metadata."""

    data: pd.DataFrame
    imputation_id: int
    method: str
    n_imputed: int
    variables_imputed: list[str]


@dataclass
class MICEResult:
    """Result of multiple imputation by chained equations.

    Parameters
    ----------
    datasets : list[ImputedDataset]
        List of *m* imputed datasets.
    n_imputations : int
        Number of imputations performed.
    n_iterations : int
        Number of MICE iterations per imputation.
    convergence : dict
        Convergence diagnostics (mean and variance traces).
    variables_imputed : list[str]
        Variables that were imputed.
    """

    datasets: list[ImputedDataset]
    n_imputations: int
    n_iterations: int
    convergence: dict
    variables_imputed: list[str]


@dataclass
class PooledEstimate:
    """Pooled estimate from Rubin's rules."""

    estimate: float
    within_variance: float
    between_variance: float
    total_variance: float
    standard_error: float
    df: float
    ci_lower: float
    ci_upper: float
    fmi: float
    relative_efficiency: float
    lambda_hat: float
    method: str = "Rubin's rules"


# ===================================================================
# PATTERN ANALYSIS
# ===================================================================


def missing_profile(data: pd.DataFrame) -> MissingProfile:
    """Comprehensive missingness profile for a DataFrame.

    Parameters
    ----------
    data : DataFrame
        Input dataset.

    Returns
    -------
    MissingProfile
    """
    n_obs, n_vars = data.shape
    total = n_obs * n_vars
    missing_mask = data.isna()
    n_missing = int(missing_mask.sum().sum())
    pct = 100 * n_missing / total if total > 0 else 0.0
    complete = int((~missing_mask).all(axis=1).sum())

    per_var = (
        pd.DataFrame(
            {
                "variable": data.columns,
                "n_missing": missing_mask.sum().values,
                "pct_missing": (100 * missing_mask.mean()).values,
                "dtype": data.dtypes.astype(str).values,
            }
        )
        .sort_values("n_missing", ascending=False)
        .reset_index(drop=True)
    )

    pattern = (~missing_mask).astype(int).values
    is_mono = _is_monotone(missing_mask.values)

    return MissingProfile(
        n_obs=n_obs,
        n_vars=n_vars,
        total_cells=total,
        n_missing=n_missing,
        pct_missing=float(pct),
        complete_cases=complete,
        per_variable=per_var,
        pattern_matrix=pattern,
        is_monotone=is_mono,
    )


def _is_monotone(missing_matrix: np.ndarray) -> bool:
    """Check whether missingness follows a monotone pattern.

    A pattern is monotone if variables can be ordered such that for all *i*,
    if variable *j* is missing then all variables *j+1, ..., p* are also missing.
    """
    # Sort columns by number of missing values
    col_missing = missing_matrix.sum(axis=0)
    order = np.argsort(col_missing)
    sorted_matrix = missing_matrix[:, order]
    n, p = sorted_matrix.shape
    for i in range(n):
        first_miss = -1
        for j in range(p):
            if sorted_matrix[i, j]:
                if first_miss == -1:
                    first_miss = j
            else:
                if first_miss != -1:
                    return False
    return True


def missing_pattern_table(data: pd.DataFrame) -> pd.DataFrame:
    """Tabulate unique missingness patterns and their frequencies.

    Parameters
    ----------
    data : DataFrame

    Returns
    -------
    DataFrame
        Each row is a unique pattern.  Columns are variable names (1 = observed,
        0 = missing) plus ``count`` and ``pct``.
    """
    pattern = (~data.isna()).astype(int)
    pattern_str = pattern.apply(lambda row: tuple(row), axis=1)
    counts = pattern_str.value_counts().reset_index()
    counts.columns = ["pattern", "count"]
    counts["pct"] = 100 * counts["count"] / len(data)
    expanded = pd.DataFrame(counts["pattern"].tolist(), columns=data.columns)
    return (
        pd.concat([expanded, counts[["count", "pct"]]], axis=1)
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


def missing_proportion_by_variable(data: pd.DataFrame) -> pd.DataFrame:
    """Return per-variable missingness proportions.

    Parameters
    ----------
    data : DataFrame

    Returns
    -------
    DataFrame
        Columns: ``variable``, ``n_missing``, ``n_observed``, ``pct_missing``.
    """
    n = len(data)
    result = []
    for col in data.columns:
        nm = int(data[col].isna().sum())
        result.append(
            {
                "variable": col,
                "n_missing": nm,
                "n_observed": n - nm,
                "pct_missing": 100 * nm / n if n > 0 else 0.0,
            }
        )
    return pd.DataFrame(result).sort_values("pct_missing", ascending=False).reset_index(drop=True)


def classify_missing_mechanism(
    data: pd.DataFrame,
    target: str,
    predictors: list[str] | None = None,
    alpha: float = 0.05,
) -> dict:
    """Heuristic classification of the missing data mechanism for a target
    variable.

    Strategy:
    - Run Little's MCAR test (if possible) or a logistic regression of the
      missingness indicator on other observed variables.
    - If the missingness is unrelated to observed variables (p > alpha), suggest
      MCAR.
    - If related to observed variables, suggest MAR.
    - MNAR cannot be tested from the data alone; a warning is issued.

    Parameters
    ----------
    data : DataFrame
    target : str
        Variable whose missingness is being classified.
    predictors : list[str] or None
        Variables to use as predictors of missingness.  If ``None``, all other
        complete columns are used.
    alpha : float, default 0.05

    Returns
    -------
    dict
        Keys: ``target``, ``classification``, ``p_value``, ``details``.
    """
    df = data.copy()
    indicator = df[target].isna().astype(int)
    if indicator.sum() == 0 or indicator.sum() == len(df):
        return {"target": target, "classification": "No missing data", "p_value": None, "details": {}}

    if predictors is None:
        # Use fully observed numeric columns
        candidates = [c for c in df.columns if c != target and df[c].notna().all()]
        predictors = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]

    if len(predictors) == 0:
        return {
            "target": target,
            "classification": "Indeterminate (no predictors)",
            "p_value": None,
            "details": {},
        }

    X = df[predictors].values.astype(np.float64)
    y = indicator.values

    try:
        model = LogisticRegression(max_iter=1000, solver="lbfgs")
        model.fit(X, y)
        from sklearn.metrics import log_loss

        ll_full = -log_loss(y, model.predict_proba(X), normalize=False)
        p_hat = y.mean()
        ll_null = len(y) * (p_hat * np.log(p_hat + 1e-15) + (1 - p_hat) * np.log(1 - p_hat + 1e-15))
        lr_stat = 2 * (ll_full - ll_null)
        dof = len(predictors)
        p_val = 1 - stats.chi2.cdf(max(lr_stat, 0), dof)
    except Exception as exc:
        logger.warning("Logistic regression failed for %s: %s", target, exc)
        return {"target": target, "classification": "Error", "p_value": None, "details": {"error": str(exc)}}

    if p_val > alpha:
        classification = "MCAR (likely)"
    else:
        classification = "MAR (likely)"

    return {
        "target": target,
        "classification": classification,
        "p_value": float(p_val),
        "details": {
            "lr_statistic": float(lr_stat),
            "df": dof,
            "predictors_used": predictors,
            "note": "MNAR cannot be ruled out from observed data alone.",
        },
    }


# ===================================================================
# LITTLE'S MCAR TEST
# ===================================================================


def littles_mcar_test(data: pd.DataFrame) -> MCARTestResult:
    """Little's MCAR test for multivariate data.

    Tests the null hypothesis that the data are missing completely at random
    by comparing observed group means (by missingness pattern) with the EM
    estimate of the overall mean.

    Parameters
    ----------
    data : DataFrame
        Numeric columns only.

    Returns
    -------
    MCARTestResult

    References
    ----------
    Little, R. J. A. (1988). A test of missing completely at random for
    multivariate data with missing values. *JASA*, 83(404), 1198--1202.
    """
    # Keep only numeric columns
    df = data.select_dtypes(include=[np.number]).copy()
    n, p = df.shape
    if p < 2:
        raise ValueError("Little's MCAR test requires at least 2 numeric variables.")

    # Identify unique missingness patterns
    pattern = (~df.isna()).astype(int)
    pattern_str = pattern.apply(lambda row: tuple(row), axis=1)
    unique_patterns = pattern_str.unique()
    n_patterns = len(unique_patterns)

    if n_patterns <= 1:
        return MCARTestResult(test_statistic=0.0, p_value=1.0, df=0, n_patterns=1)

    # Overall means and covariance from complete observations (simple EM substitute)
    # Use pairwise-complete means and covariance
    mu = df.mean().values
    sigma = df.cov().values
    sigma_reg = sigma + np.eye(p) * 1e-6  # regularise

    chi2 = 0.0
    df_test = 0

    for pat in unique_patterns:
        pat_arr = np.array(pat)
        obs_idx = np.where(pat_arr == 1)[0]
        if len(obs_idx) == 0:
            continue
        mask = pattern_str == pat
        group = df.loc[mask].iloc[:, obs_idx].dropna()
        n_j = len(group)
        if n_j <= 1:
            continue
        mu_j = group.mean().values
        mu_obs = mu[obs_idx]
        diff = mu_j - mu_obs
        sigma_obs = sigma_reg[np.ix_(obs_idx, obs_idx)]
        try:
            sigma_inv = np.linalg.inv(sigma_obs / n_j)
            chi2 += float(diff @ sigma_inv @ diff)
            df_test += len(obs_idx)
        except np.linalg.LinAlgError:
            continue

    # Adjust df for estimated parameters
    df_test = max(df_test - p, 1)
    p_val = 1 - stats.chi2.cdf(chi2, df_test)

    return MCARTestResult(
        test_statistic=float(chi2),
        p_value=float(p_val),
        df=df_test,
        n_patterns=n_patterns,
    )


# ===================================================================
# COMPLETE CASE ANALYSIS
# ===================================================================


def complete_case_analysis(
    data: pd.DataFrame,
    outcome: str | None = None,
    covariates: list[str] | None = None,
) -> dict:
    """Complete case (listwise deletion) analysis with diagnostics.

    Parameters
    ----------
    data : DataFrame
    outcome : str or None
        If provided, only these + covariate columns are checked.
    covariates : list[str] or None

    Returns
    -------
    dict
        Keys: ``complete_data``, ``n_original``, ``n_complete``,
        ``pct_dropped``, ``variables_checked``.
    """
    if outcome is not None and covariates is not None:
        cols = [outcome] + covariates
    else:
        cols = data.columns.tolist()

    n_orig = len(data)
    cc = data.dropna(subset=cols)
    n_cc = len(cc)
    pct = 100 * (1 - n_cc / n_orig) if n_orig > 0 else 0.0

    if pct > 20:
        logger.warning(
            "Complete case analysis dropped %.1f%% of observations. Consider multiple imputation.",
            pct,
        )

    return {
        "complete_data": cc,
        "n_original": n_orig,
        "n_complete": n_cc,
        "pct_dropped": float(pct),
        "variables_checked": cols,
    }


# ===================================================================
# SINGLE IMPUTATION
# ===================================================================


def impute_mean(data: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Replace missing values with the column mean.

    Parameters
    ----------
    data : DataFrame
    columns : list[str] or None
        Columns to impute.  If ``None``, all numeric columns.

    Returns
    -------
    DataFrame
    """
    df = data.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in columns:
        df[col] = df[col].fillna(df[col].mean())
    return df


def impute_median(data: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Replace missing values with the column median.

    Parameters
    ----------
    data : DataFrame
    columns : list[str] or None

    Returns
    -------
    DataFrame
    """
    df = data.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in columns:
        df[col] = df[col].fillna(df[col].median())
    return df


def impute_mode(data: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Replace missing values with the column mode.

    Parameters
    ----------
    data : DataFrame
    columns : list[str] or None

    Returns
    -------
    DataFrame
    """
    df = data.copy()
    if columns is None:
        columns = df.columns.tolist()
    for col in columns:
        mode_val = df[col].mode()
        if len(mode_val) > 0:
            df[col] = df[col].fillna(mode_val.iloc[0])
    return df


def impute_locf(data: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Last observation carried forward (LOCF) imputation.

    Parameters
    ----------
    data : DataFrame
        Assumed to be sorted in temporal order.
    columns : list[str] or None

    Returns
    -------
    DataFrame
    """
    df = data.copy()
    if columns is None:
        columns = df.columns.tolist()
    for col in columns:
        df[col] = df[col].ffill()
    return df


def impute_nocb(data: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Next observation carried backward (NOCB) imputation.

    Parameters
    ----------
    data : DataFrame
        Assumed to be sorted in temporal order.
    columns : list[str] or None

    Returns
    -------
    DataFrame
    """
    df = data.copy()
    if columns is None:
        columns = df.columns.tolist()
    for col in columns:
        df[col] = df[col].bfill()
    return df


def impute_regression(
    data: pd.DataFrame,
    target: str,
    predictors: list[str] | None = None,
    add_noise: bool = False,
) -> pd.DataFrame:
    """Regression imputation: predict missing values from a linear model.

    Parameters
    ----------
    data : DataFrame
    target : str
        Column to impute.
    predictors : list[str] or None
        Columns to use as predictors.  If ``None``, all other complete
        numeric columns.
    add_noise : bool, default False
        If ``True``, add Gaussian noise proportional to the residual standard
        deviation (stochastic regression imputation).

    Returns
    -------
    DataFrame
    """
    df = data.copy()
    if predictors is None:
        predictors = [c for c in df.select_dtypes(include=[np.number]).columns if c != target and df[c].notna().all()]
    if len(predictors) == 0:
        logger.warning("No complete predictors available; falling back to mean imputation.")
        return impute_mean(df, [target])

    observed = df.dropna(subset=[target] + predictors)
    missing = df[df[target].isna() & df[predictors].notna().all(axis=1)]

    if len(observed) < 2 or len(missing) == 0:
        return df

    X_obs = observed[predictors].values
    y_obs = observed[target].values
    model = LinearRegression().fit(X_obs, y_obs)
    X_miss = missing[predictors].values
    y_pred = model.predict(X_miss)

    if add_noise:
        residuals = y_obs - model.predict(X_obs)
        sigma = residuals.std()
        y_pred += np.random.normal(0, sigma, size=len(y_pred))

    df.loc[missing.index, target] = y_pred
    return df


def impute_hot_deck(
    data: pd.DataFrame,
    target: str,
    match_vars: list[str] | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Hot-deck imputation: replace missing values with observed values from
    a randomly selected similar donor.

    Parameters
    ----------
    data : DataFrame
    target : str
        Column to impute.
    match_vars : list[str] or None
        Variables to match on.  If ``None``, random selection from all observed.
    random_state : int or None

    Returns
    -------
    DataFrame
    """
    rng = np.random.RandomState(random_state)
    df = data.copy()
    missing_idx = df[df[target].isna()].index
    observed = df[df[target].notna()]

    if len(observed) == 0:
        logger.warning("No observed values for %s; cannot hot-deck impute.", target)
        return df

    if match_vars is None or len(match_vars) == 0:
        donors = rng.choice(observed[target].values, size=len(missing_idx), replace=True)
        df.loc[missing_idx, target] = donors
        return df

    # Match on match_vars using nearest-neighbour (simple categorical matching)
    for idx in missing_idx:
        row = df.loc[idx]
        # Find donors matching on all match_vars (exact match)
        match_mask = pd.Series(True, index=observed.index)
        for mv in match_vars:
            if pd.notna(row[mv]):
                match_mask &= observed[mv] == row[mv]
        matched = observed[match_mask]
        if len(matched) == 0:
            matched = observed  # fall back to all donors
        donor = matched.sample(1, random_state=rng.randint(0, 2**31))
        df.loc[idx, target] = donor[target].values[0]
    return df


def impute_cold_deck(
    data: pd.DataFrame,
    target: str,
    external_values: dict[Any, Any],
) -> pd.DataFrame:
    """Cold-deck imputation: replace missing values with pre-specified
    external values.

    Parameters
    ----------
    data : DataFrame
    target : str
    external_values : dict
        Mapping from index (or group key) to replacement value.  If a single
        value is desired, pass ``{None: value}``.

    Returns
    -------
    DataFrame
    """
    df = data.copy()
    missing_idx = df[df[target].isna()].index
    if None in external_values:
        df.loc[missing_idx, target] = external_values[None]
    else:
        for idx in missing_idx:
            if idx in external_values:
                df.loc[idx, target] = external_values[idx]
    return df


# ===================================================================
# MICE (Multiple Imputation by Chained Equations)
# ===================================================================


def _pmm_impute(
    y_obs: np.ndarray,
    X_obs: np.ndarray,
    X_miss: np.ndarray,
    k: int = 5,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Predictive Mean Matching imputation step.

    Fits a Bayesian-inspired linear model, draws predicted values for both
    observed and missing, then for each missing value selects from the *k*
    nearest observed predicted values and returns the corresponding observed
    outcome.
    """
    if rng is None:
        rng = np.random.RandomState()

    n_obs = len(y_obs)
    model = LinearRegression().fit(X_obs, y_obs)
    residual_var = np.var(y_obs - model.predict(X_obs))

    # Draw from posterior of beta (simplified Bayesian draw)
    beta_hat = model.coef_
    intercept = model.intercept_
    noise_scale = math.sqrt(max(residual_var, 1e-10))
    beta_draw = beta_hat + rng.normal(0, noise_scale / max(math.sqrt(n_obs), 1), size=len(beta_hat))

    pred_obs = X_obs @ beta_draw + intercept
    pred_miss = X_miss @ beta_draw + intercept

    imputed = np.empty(len(pred_miss))
    for i, pm in enumerate(pred_miss):
        distances = np.abs(pred_obs - pm)
        nearest_k = np.argsort(distances)[:k]
        donor = rng.choice(nearest_k)
        imputed[i] = y_obs[donor]
    return imputed


def _bayesian_linear_impute(
    y_obs: np.ndarray,
    X_obs: np.ndarray,
    X_miss: np.ndarray,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Bayesian linear regression imputation step."""
    if rng is None:
        rng = np.random.RandomState()
    model = BayesianRidge()
    model.fit(X_obs, y_obs)
    y_pred, y_std = model.predict(X_miss, return_std=True)
    return y_pred + rng.normal(0, y_std)


def _logistic_impute(
    y_obs: np.ndarray,
    X_obs: np.ndarray,
    X_miss: np.ndarray,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Logistic regression imputation for binary variables."""
    if rng is None:
        rng = np.random.RandomState()
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_obs, y_obs.astype(int))
    probs = model.predict_proba(X_miss)[:, 1]
    return (rng.random(len(probs)) < probs).astype(float)


def _polytomous_impute(
    y_obs: np.ndarray,
    X_obs: np.ndarray,
    X_miss: np.ndarray,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Polytomous (multinomial) logistic regression imputation for
    categorical variables."""
    if rng is None:
        rng = np.random.RandomState()
    model = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="multinomial")
    model.fit(X_obs, y_obs.astype(int))
    probs = model.predict_proba(X_miss)
    classes = model.classes_
    imputed = np.empty(len(probs))
    for i, p in enumerate(probs):
        imputed[i] = rng.choice(classes, p=p)
    return imputed


def _rf_impute(
    y_obs: np.ndarray,
    X_obs: np.ndarray,
    X_miss: np.ndarray,
    is_categorical: bool = False,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Random forest imputation step."""
    if rng is None:
        rng = np.random.RandomState()
    if is_categorical:
        model = RandomForestClassifier(n_estimators=50, random_state=rng.randint(0, 2**31))
        model.fit(X_obs, y_obs.astype(int))
        # Draw from leaf-level predictions for variability
        preds = model.predict(X_miss)
    else:
        model = RandomForestRegressor(n_estimators=50, random_state=rng.randint(0, 2**31))
        model.fit(X_obs, y_obs)
        # Use individual tree predictions for noise
        tree_preds = np.array([tree.predict(X_miss) for tree in model.estimators_])
        # Sample one tree per observation
        preds = np.array([rng.choice(tree_preds[:, i]) for i in range(X_miss.shape[0])])
    return preds


def mice(
    data: pd.DataFrame,
    m: int = 5,
    max_iter: int = 10,
    method: Union[str, dict[str, str]] = "pmm",
    random_state: int | None = None,
    pmm_k: int = 5,
) -> MICEResult:
    """Multiple Imputation by Chained Equations (MICE).

    Parameters
    ----------
    data : DataFrame
        Dataset with missing values.
    m : int, default 5
        Number of imputations.
    max_iter : int, default 10
        Number of MICE iterations (cycles through all variables).
    method : str or dict
        Imputation method per variable or a single method for all:
        ``"pmm"`` (predictive mean matching), ``"bayesian"`` (Bayesian linear),
        ``"logistic"`` (binary), ``"polytomous"`` (categorical), ``"rf"``
        (random forest).
    random_state : int or None
        Seed for reproducibility.
    pmm_k : int, default 5
        Number of nearest neighbours for PMM.

    Returns
    -------
    MICEResult

    References
    ----------
    van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate
    Imputation by Chained Equations in R. *Journal of Statistical Software*,
    45(3), 1--67.
    """
    rng = np.random.RandomState(random_state)
    df = data.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    vars_with_missing = [c for c in df.columns if df[c].isna().any()]

    if len(vars_with_missing) == 0:
        logger.info("No missing values found; returning original data.")
        return MICEResult(
            datasets=[ImputedDataset(df.copy(), 0, "none", 0, [])],
            n_imputations=1,
            n_iterations=0,
            convergence={},
            variables_imputed=[],
        )

    # Determine method per variable
    if isinstance(method, str):
        method_map = {v: method for v in vars_with_missing}
    else:
        method_map = method

    # Trace storage for convergence diagnostics
    traces: dict[str, dict[str, list[float]]] = {v: {"mean": [], "var": []} for v in vars_with_missing}

    datasets = []

    for imp_id in range(m):
        imp_rng = np.random.RandomState(rng.randint(0, 2**31))
        # Initialise with mean/mode
        filled = df.copy()
        for col in vars_with_missing:
            if col in numeric_cols:
                filled[col] = filled[col].fillna(filled[col].mean())
            else:
                mode = filled[col].mode()
                if len(mode) > 0:
                    filled[col] = filled[col].fillna(mode.iloc[0])

        for iteration in range(max_iter):
            for target in vars_with_missing:
                missing_mask = df[target].isna()
                if missing_mask.sum() == 0:
                    continue

                predictors = [c for c in numeric_cols if c != target]
                if len(predictors) == 0:
                    continue

                # Observed and missing splits
                obs_mask = ~missing_mask
                X_obs = filled.loc[obs_mask, predictors].values.astype(np.float64)
                y_obs = filled.loc[obs_mask, target].values.astype(np.float64)
                X_miss = filled.loc[missing_mask, predictors].values.astype(np.float64)

                if len(X_obs) < 2 or len(X_miss) == 0:
                    continue

                m_method = method_map.get(target, "pmm")

                if m_method == "pmm":
                    imputed_vals = _pmm_impute(y_obs, X_obs, X_miss, k=pmm_k, rng=imp_rng)
                elif m_method == "bayesian":
                    imputed_vals = _bayesian_linear_impute(y_obs, X_obs, X_miss, rng=imp_rng)
                elif m_method == "logistic":
                    imputed_vals = _logistic_impute(y_obs, X_obs, X_miss, rng=imp_rng)
                elif m_method == "polytomous":
                    imputed_vals = _polytomous_impute(y_obs, X_obs, X_miss, rng=imp_rng)
                elif m_method == "rf":
                    is_cat = target not in numeric_cols
                    imputed_vals = _rf_impute(y_obs, X_obs, X_miss, is_categorical=is_cat, rng=imp_rng)
                else:
                    raise ValueError(f"Unknown imputation method: {m_method}")

                filled.loc[missing_mask, target] = imputed_vals

                # Track convergence
                if target in traces:
                    traces[target]["mean"].append(float(filled[target].mean()))
                    traces[target]["var"].append(float(filled[target].var()))

        n_imputed = int(df.isna().sum().sum())
        datasets.append(
            ImputedDataset(
                data=filled,
                imputation_id=imp_id,
                method=str(method),
                n_imputed=n_imputed,
                variables_imputed=vars_with_missing,
            )
        )

    return MICEResult(
        datasets=datasets,
        n_imputations=m,
        n_iterations=max_iter,
        convergence=traces,
        variables_imputed=vars_with_missing,
    )


# ===================================================================
# RUBIN'S RULES
# ===================================================================


def rubins_rules(
    estimates: Union[np.ndarray, list[float]],
    variances: Union[np.ndarray, list[float]],
    confidence: float = 0.95,
) -> PooledEstimate:
    """Pool multiply imputed estimates using Rubin's rules.

    Parameters
    ----------
    estimates : array-like
        Point estimates from each imputed dataset.
    variances : array-like
        Variance estimates from each imputed dataset.
    confidence : float, default 0.95

    Returns
    -------
    PooledEstimate

    References
    ----------
    Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*.
    Wiley. Chapter 3.
    """
    Q = np.asarray(estimates, dtype=np.float64)
    U = np.asarray(variances, dtype=np.float64)
    m = len(Q)
    if m < 2:
        raise ValueError("Need at least 2 imputations for pooling.")

    Q_bar = float(Q.mean())
    U_bar = float(U.mean())  # within-imputation variance
    B = float(Q.var(ddof=1))  # between-imputation variance
    T = U_bar + (1 + 1 / m) * B  # total variance
    se = math.sqrt(T)

    # Fraction of missing information
    r = (1 + 1 / m) * B / U_bar if U_bar > 0 else 0.0
    lambda_hat = (1 + 1 / m) * B / T if T > 0 else 0.0
    fmi = (r + 2 / (m * (1 - lambda_hat) + 3)) / (r + 1) if (r + 1) > 0 else 0.0

    # Degrees of freedom (Barnard & Rubin, 1999)
    v_old = (m - 1) * (1 + 1 / r) ** 2 if r > 0 else float("inf")
    # Adjusted df for small samples: use v_old as approximation
    df_val = max(v_old, 1.0)

    # Relative efficiency
    re = 1 / (1 + lambda_hat / m) if m > 0 else 1.0

    t_crit = stats.t.ppf((1 + confidence) / 2, df_val)
    ci_lo = Q_bar - t_crit * se
    ci_hi = Q_bar + t_crit * se

    return PooledEstimate(
        estimate=Q_bar,
        within_variance=U_bar,
        between_variance=B,
        total_variance=T,
        standard_error=se,
        df=df_val,
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        fmi=float(fmi),
        relative_efficiency=float(re),
        lambda_hat=float(lambda_hat),
    )


def pool_mice_estimates(
    mice_result: MICEResult,
    analysis_func: Callable[[pd.DataFrame], tuple[float, float]],
    confidence: float = 0.95,
) -> PooledEstimate:
    """Apply an analysis function to each imputed dataset and pool with
    Rubin's rules.

    Parameters
    ----------
    mice_result : MICEResult
        Output from :func:`mice`.
    analysis_func : callable
        Function that takes a DataFrame and returns ``(estimate, variance)``.
    confidence : float, default 0.95

    Returns
    -------
    PooledEstimate
    """
    estimates = []
    variances = []
    for ds in mice_result.datasets:
        est, var = analysis_func(ds.data)
        estimates.append(est)
        variances.append(var)
    return rubins_rules(estimates, variances, confidence=confidence)


# ===================================================================
# CONVERGENCE DIAGNOSTICS
# ===================================================================


def mice_convergence_data(mice_result: MICEResult) -> pd.DataFrame:
    """Extract convergence trace data from a MICE result for plotting.

    Parameters
    ----------
    mice_result : MICEResult

    Returns
    -------
    DataFrame
        Columns: ``variable``, ``iteration``, ``mean``, ``variance``.
    """
    rows = []
    for var, traces in mice_result.convergence.items():
        for i, (m, v) in enumerate(zip(traces["mean"], traces["var"])):
            rows.append(
                {
                    "variable": var,
                    "iteration": i + 1,
                    "mean": m,
                    "variance": v,
                }
            )
    return pd.DataFrame(rows)


def fraction_missing_information(
    estimates: Union[np.ndarray, list[float]],
    variances: Union[np.ndarray, list[float]],
) -> float:
    """Compute the fraction of missing information (FMI).

    Parameters
    ----------
    estimates : array-like
        Point estimates from each imputed dataset.
    variances : array-like
        Within-imputation variances.

    Returns
    -------
    float
        FMI value in [0, 1].
    """
    result = rubins_rules(estimates, variances)
    return result.fmi


def relative_efficiency(m: int, fmi: float) -> float:
    """Compute the relative efficiency of *m* imputations.

    :math:`RE = 1 / (1 + \\gamma / m)` where :math:`\\gamma` is the FMI.

    Parameters
    ----------
    m : int
        Number of imputations.
    fmi : float
        Fraction of missing information.

    Returns
    -------
    float
    """
    return 1.0 / (1.0 + fmi / m) if m > 0 else 0.0


# ===================================================================
# SENSITIVITY ANALYSIS
# ===================================================================


def tipping_point_analysis(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    covariates: list[str],
    mice_result: MICEResult,
    delta_range: Sequence[float] | None = None,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Tipping-point sensitivity analysis for MNAR.

    Shifts imputed values of the outcome by a sequence of delta values
    and re-analyses to find the delta at which the treatment effect becomes
    non-significant.

    Parameters
    ----------
    data : DataFrame
        Original data with missing values.
    outcome : str
        Outcome variable name.
    treatment : str
        Treatment variable name.
    covariates : list[str]
        Covariate names.
    mice_result : MICEResult
        Fitted MICE result.
    delta_range : sequence of float or None
        Deltas to test.  If ``None``, uses ``np.linspace(-2, 2, 21)``.
    confidence : float, default 0.95

    Returns
    -------
    DataFrame
        Columns: ``delta``, ``estimate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``p_value``, ``significant``.
    """
    import statsmodels.formula.api as smf

    if delta_range is None:
        delta_range = np.linspace(-2, 2, 21)

    original_missing = data[outcome].isna()
    results = []

    for delta in delta_range:
        estimates = []
        variances_list = []
        for ds in mice_result.datasets:
            shifted = ds.data.copy()
            shifted.loc[original_missing, outcome] = shifted.loc[original_missing, outcome] + delta
            formula = f"{outcome} ~ {treatment} + " + " + ".join(covariates)
            try:
                model = smf.ols(formula, data=shifted).fit()
                est = float(model.params.get(treatment, 0))
                var = float(model.bse.get(treatment, 0) ** 2)
                estimates.append(est)
                variances_list.append(var)
            except Exception:
                continue

        if len(estimates) < 2:
            continue

        pooled = rubins_rules(estimates, variances_list, confidence=confidence)
        p_val = 2 * stats.t.sf(abs(pooled.estimate / pooled.standard_error), pooled.df)
        results.append(
            {
                "delta": float(delta),
                "estimate": pooled.estimate,
                "se": pooled.standard_error,
                "ci_lower": pooled.ci_lower,
                "ci_upper": pooled.ci_upper,
                "p_value": float(p_val),
                "significant": float(p_val) < (1 - confidence),
            }
        )

    return pd.DataFrame(results)


def delta_adjustment(
    mice_result: MICEResult,
    variable: str,
    delta: float,
    group_col: str | None = None,
    group_value: Any | None = None,
    original_data: pd.DataFrame | None = None,
) -> MICEResult:
    """Apply a delta adjustment to imputed values for sensitivity analysis.

    Adds *delta* to the imputed values of *variable* for observations that
    were originally missing.

    Parameters
    ----------
    mice_result : MICEResult
    variable : str
    delta : float
    group_col : str or None
        If provided, only adjust within this group.
    group_value : any
        Value of group_col to restrict adjustment to.
    original_data : DataFrame or None
        Original data to identify which values were imputed.

    Returns
    -------
    MICEResult
        New MICEResult with adjusted datasets.
    """
    new_datasets = []
    for ds in mice_result.datasets:
        adjusted = ds.data.copy()
        if original_data is not None:
            was_missing = original_data[variable].isna()
        else:
            was_missing = pd.Series(False, index=adjusted.index)
            logger.warning("No original_data provided; adjusting all values.")
            was_missing[:] = True

        if group_col is not None and group_value is not None:
            was_missing = was_missing & (adjusted[group_col] == group_value)

        adjusted.loc[was_missing, variable] = adjusted.loc[was_missing, variable] + delta
        new_datasets.append(
            ImputedDataset(
                data=adjusted,
                imputation_id=ds.imputation_id,
                method=f"{ds.method}+delta({delta})",
                n_imputed=ds.n_imputed,
                variables_imputed=ds.variables_imputed,
            )
        )
    return MICEResult(
        datasets=new_datasets,
        n_imputations=mice_result.n_imputations,
        n_iterations=mice_result.n_iterations,
        convergence=mice_result.convergence,
        variables_imputed=mice_result.variables_imputed,
    )


# ===================================================================
# AUXILIARY VARIABLE SELECTION
# ===================================================================


def select_auxiliary_variables(
    data: pd.DataFrame,
    target: str,
    r_threshold: float = 0.1,
    max_vars: int = 20,
) -> list[str]:
    """Select auxiliary variables for imputation model.

    Auxiliary variables are those that predict the target or its missingness
    indicator and should be included in the imputation model even if they are
    not in the analysis model.

    Strategy: Compute the absolute correlation of each observed variable with
    (1) the target (among complete cases) and (2) the missingness indicator.
    Variables exceeding ``r_threshold`` in either are selected.

    Parameters
    ----------
    data : DataFrame
    target : str
    r_threshold : float, default 0.1
        Minimum absolute correlation to qualify.
    max_vars : int, default 20
        Maximum number of auxiliary variables.

    Returns
    -------
    list[str]
    """
    df = data.copy()
    indicator = df[target].isna().astype(float)
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]

    scores = []
    for col in numeric_cols:
        valid = df[[col, target]].dropna()
        if len(valid) > 2:
            r_target = abs(valid[col].corr(valid[target]))
        else:
            r_target = 0.0
        valid_ind = df[col].dropna()
        common = valid_ind.index.intersection(indicator.index)
        if len(common) > 2:
            r_miss = abs(valid_ind.loc[common].corr(indicator.loc[common]))
        else:
            r_miss = 0.0
        max_r = max(r_target, r_miss)
        if max_r >= r_threshold:
            scores.append((col, max_r))

    scores.sort(key=lambda x: -x[1])
    return [s[0] for s in scores[:max_vars]]


# ===================================================================
# PASSIVE IMPUTATION
# ===================================================================


def passive_impute(
    data: pd.DataFrame,
    derived_col: str,
    formula_func: Callable[[pd.DataFrame], pd.Series],
) -> pd.DataFrame:
    """Passive imputation for derived variables.

    After imputing constituent variables, recompute the derived variable
    using the provided formula function.

    Parameters
    ----------
    data : DataFrame
    derived_col : str
        Name of the derived column.
    formula_func : callable
        Function that takes the DataFrame and returns a Series of computed
        values for the derived column.

    Returns
    -------
    DataFrame
    """
    df = data.copy()
    df[derived_col] = formula_func(df)
    return df


# ===================================================================
# POST-IMPUTATION DIAGNOSTICS
# ===================================================================


def imputation_diagnostics(
    original: pd.DataFrame,
    imputed: pd.DataFrame,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """Compare distributions of observed and imputed values.

    Parameters
    ----------
    original : DataFrame
        Original dataset with missing values.
    imputed : DataFrame
        Imputed dataset.
    variables : list[str] or None
        Variables to check.  If ``None``, all variables that had missing data.

    Returns
    -------
    DataFrame
        Columns: ``variable``, ``obs_mean``, ``imp_mean``, ``obs_sd``,
        ``imp_sd``, ``obs_min``, ``imp_min``, ``obs_max``, ``imp_max``,
        ``ks_statistic``, ``ks_p_value``.
    """
    if variables is None:
        variables = [c for c in original.columns if original[c].isna().any()]

    results = []
    for var in variables:
        if var not in original.columns or var not in imputed.columns:
            continue
        if not pd.api.types.is_numeric_dtype(original[var]):
            continue

        obs = original[var].dropna().values.astype(np.float64)
        was_missing = original[var].isna()
        imp = imputed.loc[was_missing, var].values.astype(np.float64)

        if len(obs) == 0 or len(imp) == 0:
            continue

        ks_stat, ks_p = stats.ks_2samp(obs, imp)
        results.append(
            {
                "variable": var,
                "obs_mean": float(obs.mean()),
                "imp_mean": float(imp.mean()),
                "obs_sd": float(obs.std(ddof=1)),
                "imp_sd": float(imp.std(ddof=1)) if len(imp) > 1 else 0.0,
                "obs_min": float(obs.min()),
                "imp_min": float(imp.min()),
                "obs_max": float(obs.max()),
                "imp_max": float(imp.max()),
                "ks_statistic": float(ks_stat),
                "ks_p_value": float(ks_p),
            }
        )
    return pd.DataFrame(results)


def plot_imputation_density_data(
    original: pd.DataFrame,
    imputed_datasets: list[pd.DataFrame],
    variable: str,
) -> pd.DataFrame:
    """Generate data for overlaid density plots of observed vs. imputed values.

    Parameters
    ----------
    original : DataFrame
    imputed_datasets : list[DataFrame]
        Multiple imputed datasets.
    variable : str

    Returns
    -------
    DataFrame
        Columns: ``value``, ``source``, ``imputation_id``.
    """
    obs = original[variable].dropna()
    frames = [pd.DataFrame({"value": obs, "source": "observed", "imputation_id": 0})]
    was_missing = original[variable].isna()
    for i, imp_df in enumerate(imputed_datasets):
        imp_vals = imp_df.loc[was_missing, variable].dropna()
        frames.append(
            pd.DataFrame(
                {
                    "value": imp_vals,
                    "source": "imputed",
                    "imputation_id": i + 1,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
