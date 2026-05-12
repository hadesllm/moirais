"""
Data and model validation framework for epidemiological analysis.

Provides schema validation, data quality scoring, cross-validation strategies,
calibration assessment, discrimination metrics, decision curve analysis,
overfitting detection, and temporal/external validation frameworks.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.base import BaseEstimator, clone

logger = logging.getLogger(__name__)

# ===========================================================================
# Schema validation
# ===========================================================================


@dataclass
class ColumnRule:
    """Validation rule for a single column.

    Parameters
    ----------
    name : str
        Column name.
    dtype : str or None
        Expected pandas dtype family (e.g. ``"numeric"``, ``"object"``,
        ``"datetime"``).
    required : bool
        Whether the column must be present.
    nullable : bool
        Whether null values are allowed.
    null_threshold : float
        Maximum fraction of nulls allowed (0.0 -- 1.0).
    min_val : float or None
        Minimum allowed value (numeric columns).
    max_val : float or None
        Maximum allowed value.
    allowed_values : set or None
        If provided, all non-null values must be in this set.
    unique : bool
        Whether all values must be unique.
    regex : str or None
        A regex pattern that all string values must match.
    custom : Callable or None
        A lambda/function ``(pd.Series) -> bool``.  Returns ``True``
        if the column passes.
    """

    name: str
    dtype: str | None = None
    required: bool = True
    nullable: bool = True
    null_threshold: float = 1.0
    min_val: float | None = None
    max_val: float | None = None
    allowed_values: set | None = None
    unique: bool = False
    regex: str | None = None
    custom: Callable[[pd.Series], bool] | None = None


@dataclass
class SchemaValidationResult:
    """Result container for schema validation."""

    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_schema(
    data: pd.DataFrame,
    rules: list[ColumnRule],
    *,
    raise_on_error: bool = False,
) -> SchemaValidationResult:
    """
    Validate a DataFrame against a schema defined by column rules.

    Parameters
    ----------
    data : DataFrame
        Data to validate.
    rules : list of ColumnRule
        Validation rules for each column.
    raise_on_error : bool
        If ``True``, raise ``ValueError`` on the first error.

    Returns
    -------
    SchemaValidationResult
    """
    result = SchemaValidationResult(passed=True)

    for rule in rules:
        # Presence check
        if rule.name not in data.columns:
            if rule.required:
                msg = f"Missing required column: '{rule.name}'"
                result.errors.append(msg)
                result.passed = False
                if raise_on_error:
                    raise ValueError(msg)
            continue

        col = data[rule.name]

        # Dtype check
        if rule.dtype is not None:
            if rule.dtype == "numeric" and not pd.api.types.is_numeric_dtype(col):
                msg = f"Column '{rule.name}' expected numeric, got {col.dtype}"
                result.errors.append(msg)
                result.passed = False
            elif rule.dtype == "object" and not pd.api.types.is_string_dtype(col):
                # Allow category as well
                if not pd.api.types.is_categorical_dtype(col):
                    result.warnings.append(f"Column '{rule.name}' expected string/object, got {col.dtype}")
            elif rule.dtype == "datetime" and not pd.api.types.is_datetime64_any_dtype(col):
                msg = f"Column '{rule.name}' expected datetime, got {col.dtype}"
                result.errors.append(msg)
                result.passed = False

        # Null checks
        null_frac = col.isna().mean()
        if not rule.nullable and null_frac > 0:
            msg = f"Column '{rule.name}' has {col.isna().sum()} nulls but nullable=False"
            result.errors.append(msg)
            result.passed = False
        if null_frac > rule.null_threshold:
            msg = f"Column '{rule.name}' null fraction {null_frac:.3f} exceeds threshold {rule.null_threshold}"
            result.errors.append(msg)
            result.passed = False

        # Value range
        if rule.min_val is not None and pd.api.types.is_numeric_dtype(col):
            violations = (col.dropna() < rule.min_val).sum()
            if violations > 0:
                msg = f"Column '{rule.name}': {violations} values below min={rule.min_val}"
                result.errors.append(msg)
                result.passed = False

        if rule.max_val is not None and pd.api.types.is_numeric_dtype(col):
            violations = (col.dropna() > rule.max_val).sum()
            if violations > 0:
                msg = f"Column '{rule.name}': {violations} values above max={rule.max_val}"
                result.errors.append(msg)
                result.passed = False

        # Allowed values
        if rule.allowed_values is not None:
            invalid = col.dropna()[~col.dropna().isin(rule.allowed_values)]
            if len(invalid) > 0:
                bad_vals = invalid.unique()[:5]
                msg = f"Column '{rule.name}': {len(invalid)} values not in allowed set. Examples: {list(bad_vals)}"
                result.errors.append(msg)
                result.passed = False

        # Uniqueness
        if rule.unique:
            n_dup = col.dropna().duplicated().sum()
            if n_dup > 0:
                msg = f"Column '{rule.name}' has {n_dup} duplicate values (unique=True)"
                result.errors.append(msg)
                result.passed = False

        # Regex
        if rule.regex is not None and pd.api.types.is_string_dtype(col):
            import re

            pattern = re.compile(rule.regex)
            non_null = col.dropna()
            non_match = non_null[~non_null.str.match(pattern)].count()
            if non_match > 0:
                msg = f"Column '{rule.name}': {non_match} values don't match regex '{rule.regex}'"
                result.errors.append(msg)
                result.passed = False

        # Custom
        if rule.custom is not None:
            try:
                if not rule.custom(col):
                    msg = f"Column '{rule.name}': custom validation failed"
                    result.errors.append(msg)
                    result.passed = False
            except Exception as exc:
                msg = f"Column '{rule.name}': custom validation raised {type(exc).__name__}: {exc}"
                result.errors.append(msg)
                result.passed = False

    if raise_on_error and not result.passed:
        raise ValueError(f"Schema validation failed: {result.errors}")

    return result


def check_referential_integrity(
    child: pd.DataFrame,
    parent: pd.DataFrame,
    child_key: str,
    parent_key: str,
) -> SchemaValidationResult:
    """
    Check that all values in *child_key* appear in *parent_key*.

    Parameters
    ----------
    child : DataFrame
        Table with the foreign key.
    parent : DataFrame
        Table with the primary key.
    child_key : str
        Column in *child*.
    parent_key : str
        Column in *parent*.

    Returns
    -------
    SchemaValidationResult
    """
    result = SchemaValidationResult(passed=True)
    parent_vals = set(parent[parent_key].dropna())
    child_vals = child[child_key].dropna()
    orphans = child_vals[~child_vals.isin(parent_vals)]
    if len(orphans) > 0:
        result.passed = False
        examples = orphans.unique()[:5]
        result.errors.append(
            f"{len(orphans)} orphan rows in '{child_key}' not found in '{parent_key}'. Examples: {list(examples)}"
        )
    return result


# ===========================================================================
# Data quality scoring
# ===========================================================================


@dataclass
class DataQualityReport:
    """Data quality assessment scores."""

    completeness: float
    consistency: float
    timeliness: float
    uniqueness: float
    overall: float
    details: dict[str, Any] = field(default_factory=dict)


def score_data_quality(
    data: pd.DataFrame,
    *,
    date_cols: list[str] | None = None,
    freshness_days: int = 365,
    key_cols: list[str] | None = None,
    consistency_rules: list[Callable[[pd.DataFrame], bool]] | None = None,
) -> DataQualityReport:
    """
    Compute a multi-dimensional data quality score.

    Parameters
    ----------
    data : DataFrame
        Input data.
    date_cols : list of str or None
        Datetime columns used for timeliness scoring.
    freshness_days : int
        Maximum age in days for full timeliness score.
    key_cols : list of str or None
        Columns that should be unique together (for uniqueness score).
    consistency_rules : list of callable or None
        Each callable takes the DataFrame and returns ``True`` if the
        consistency check passes.

    Returns
    -------
    DataQualityReport
        Scores between 0.0 and 1.0 for each dimension and an overall
        composite score.
    """
    details: dict[str, Any] = {}

    # Completeness: fraction of non-null cells
    total_cells = data.shape[0] * data.shape[1]
    non_null = data.notna().sum().sum()
    completeness = float(non_null / total_cells) if total_cells > 0 else 0.0
    details["null_counts"] = data.isna().sum().to_dict()

    # Consistency: fraction of rules that pass
    if consistency_rules:
        n_pass = sum(1 for fn in consistency_rules if fn(data))
        consistency = n_pass / len(consistency_rules)
        details["consistency_checks"] = len(consistency_rules)
        details["consistency_passed"] = n_pass
    else:
        consistency = 1.0
        details["consistency_checks"] = 0

    # Timeliness: how fresh are date columns
    if date_cols:
        freshness_scores = []
        now = pd.Timestamp.now()
        for dc in date_cols:
            if dc in data.columns and pd.api.types.is_datetime64_any_dtype(data[dc]):
                max_date = data[dc].max()
                if pd.notna(max_date):
                    age_days = (now - max_date).days
                    score = max(0.0, 1.0 - age_days / freshness_days)
                    freshness_scores.append(score)
                    details[f"timeliness_{dc}"] = {"max_date": str(max_date), "age_days": age_days}
        timeliness = float(np.mean(freshness_scores)) if freshness_scores else 1.0
    else:
        timeliness = 1.0

    # Uniqueness: duplicate rate in key columns
    if key_cols:
        subset = data[key_cols].dropna()
        n_dup = subset.duplicated().sum()
        uniqueness = 1.0 - (n_dup / len(subset)) if len(subset) > 0 else 1.0
        details["key_duplicates"] = int(n_dup)
    else:
        n_dup = data.duplicated().sum()
        uniqueness = 1.0 - (n_dup / len(data)) if len(data) > 0 else 1.0
        details["row_duplicates"] = int(n_dup)

    overall = float(np.mean([completeness, consistency, timeliness, uniqueness]))

    return DataQualityReport(
        completeness=completeness,
        consistency=consistency,
        timeliness=timeliness,
        uniqueness=uniqueness,
        overall=overall,
        details=details,
    )


# ===========================================================================
# Cross-validation strategies
# ===========================================================================


@dataclass
class CVResult:
    """Cross-validation result."""

    scores: np.ndarray
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    fold_details: list[dict[str, Any]] = field(default_factory=list)


def cross_validate(
    estimator: BaseEstimator,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    method: str = "stratified_kfold",
    n_folds: int = 5,
    n_repeats: int = 10,
    scoring: str = "roc_auc",
    groups: np.ndarray | pd.Series | None = None,
    confidence: float = 0.95,
    random_state: int = 42,
) -> CVResult:
    """
    Cross-validate a model using various strategies.

    Parameters
    ----------
    estimator : sklearn estimator
        An unfitted estimator (will be cloned per fold).
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    method : str
        Cross-validation strategy:
        ``"kfold"``, ``"stratified_kfold"``, ``"grouped_kfold"``,
        ``"loo"``, ``"monte_carlo"``, ``"time_series"``,
        ``"nested"`` (outer only -- see :func:`nested_cross_validate`).
    n_folds : int
        Number of folds (ignored for LOO).
    n_repeats : int
        Number of repeats (for ``"monte_carlo"``).
    scoring : str
        Scoring metric name (scikit-learn convention).
    groups : array-like or None
        Group labels for grouped k-fold.
    confidence : float
        Confidence level for the CI on mean score.
    random_state : int
        Random seed.

    Returns
    -------
    CVResult
    """
    from sklearn.model_selection import (
        GroupKFold,
        KFold,
        LeaveOneOut,
        ShuffleSplit,
        StratifiedKFold,
        TimeSeriesSplit,
        cross_val_score,
    )

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    if method == "kfold":
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    elif method == "stratified_kfold":
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    elif method == "grouped_kfold":
        if groups is None:
            raise ValueError("groups must be provided for grouped_kfold")
        cv = GroupKFold(n_splits=n_folds)
    elif method == "loo":
        cv = LeaveOneOut()
    elif method == "monte_carlo":
        cv = ShuffleSplit(n_splits=n_repeats, test_size=1.0 / n_folds, random_state=random_state)
    elif method == "time_series":
        cv = TimeSeriesSplit(n_splits=n_folds)
    else:
        raise ValueError(f"Unknown CV method: {method}")

    scores = cross_val_score(estimator, X_arr, y_arr, cv=cv, scoring=scoring, groups=groups)

    z = sp_stats.norm.ppf(1 - (1 - confidence) / 2)
    se = scores.std() / np.sqrt(len(scores))

    return CVResult(
        scores=scores,
        mean=float(scores.mean()),
        std=float(scores.std()),
        ci_lower=float(scores.mean() - z * se),
        ci_upper=float(scores.mean() + z * se),
    )


def nested_cross_validate(
    estimator: BaseEstimator,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    param_grid: dict[str, list[Any]],
    outer_folds: int = 5,
    inner_folds: int = 3,
    scoring: str = "roc_auc",
    random_state: int = 42,
) -> CVResult:
    """
    Nested cross-validation for unbiased performance estimation.

    Parameters
    ----------
    estimator : sklearn estimator
    X : array-like
    y : array-like
    param_grid : dict
        Hyperparameter grid for inner-loop search.
    outer_folds : int
    inner_folds : int
    scoring : str
    random_state : int

    Returns
    -------
    CVResult
    """
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)

    gs = GridSearchCV(estimator, param_grid, cv=inner_cv, scoring=scoring, refit=True)
    scores = cross_val_score(gs, np.asarray(X), np.asarray(y), cv=outer_cv, scoring=scoring)

    z = sp_stats.norm.ppf(0.975)
    se = scores.std() / np.sqrt(len(scores))

    return CVResult(
        scores=scores,
        mean=float(scores.mean()),
        std=float(scores.std()),
        ci_lower=float(scores.mean() - z * se),
        ci_upper=float(scores.mean() + z * se),
    )


def bootstrap_validate(
    estimator: BaseEstimator,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    n_bootstraps: int = 200,
    scoring: str = "roc_auc",
    method: str = "632plus",
    random_state: int = 42,
) -> CVResult:
    """
    Bootstrap validation with .632 or .632+ correction.

    Parameters
    ----------
    estimator : sklearn estimator
    X : array-like
    y : array-like
    n_bootstraps : int
    scoring : str
        ``"roc_auc"`` or ``"accuracy"``.
    method : str
        ``"632"`` or ``"632plus"``.
    random_state : int

    Returns
    -------
    CVResult

    References
    ----------
    Efron, B., & Tibshirani, R. (1997). Improvements on cross-validation:
    the .632+ bootstrap method. *Journal of the American Statistical
    Association*, 92(438), 548-560.
    https://doi.org/10.1080/01621459.1997.10474007
    """
    from sklearn.metrics import get_scorer

    scorer = get_scorer(scoring)
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    n = len(y_arr)
    rng = np.random.default_rng(random_state)

    # Apparent performance
    est_app = clone(estimator).fit(X_arr, y_arr)
    apparent = scorer(est_app, X_arr, y_arr)

    boot_scores = []
    oob_scores = []

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        oob_idx = np.setdiff1d(np.arange(n), idx)

        if len(oob_idx) == 0 or len(np.unique(y_arr[idx])) < 2:
            continue

        est_b = clone(estimator).fit(X_arr[idx], y_arr[idx])
        oob_score = scorer(est_b, X_arr[oob_idx], y_arr[oob_idx])
        boot_score = scorer(est_b, X_arr[idx], y_arr[idx])

        boot_scores.append(boot_score)
        oob_scores.append(oob_score)

    oob_arr = np.array(oob_scores)
    boot_arr = np.array(boot_scores)

    if method == "632":
        corrected = 0.368 * apparent + 0.632 * oob_arr.mean()
    elif method == "632plus":
        # No-information rate
        y_pred_random = rng.permutation(y_arr)
        est_ni = clone(estimator).fit(X_arr, y_arr)
        gamma = scorer(est_ni, X_arr, y_pred_random)
        # Relative overfit rate
        r_bar = (oob_arr.mean() - apparent) / (gamma - apparent) if abs(gamma - apparent) > 1e-12 else 0
        r_bar = np.clip(r_bar, 0, 1)
        w = 0.632 / (1 - 0.368 * r_bar)
        corrected = (1 - w) * apparent + w * oob_arr.mean()
    else:
        raise ValueError(f"Unknown bootstrap method: {method}. Use '632' or '632plus'.")

    scores_arr = np.array([corrected] + list(oob_arr))
    z = sp_stats.norm.ppf(0.975)
    se = oob_arr.std() / np.sqrt(len(oob_arr))

    return CVResult(
        scores=oob_arr,
        mean=float(corrected),
        std=float(oob_arr.std()),
        ci_lower=float(corrected - z * se),
        ci_upper=float(corrected + z * se),
    )


# ===========================================================================
# Calibration assessment
# ===========================================================================


@dataclass
class CalibrationResult:
    """Calibration assessment metrics."""

    hosmer_lemeshow_stat: float
    hosmer_lemeshow_p: float
    calibration_slope: float
    calibration_intercept: float
    brier_score: float
    scaled_brier: float
    calibration_in_the_large: float


def assess_calibration(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    n_groups: int = 10,
) -> CalibrationResult:
    """
    Comprehensive calibration assessment for a binary outcome model.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted probabilities.
    n_groups : int
        Number of groups for the Hosmer-Lemeshow test.

    Returns
    -------
    CalibrationResult

    References
    ----------
    Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013).
    *Applied Logistic Regression* (3rd ed.). Wiley.
    https://doi.org/10.1002/9781118548387

    Steyerberg, E. W., et al. (2010). Assessing the performance of
    prediction models: a framework for traditional and novel measures.
    *Epidemiology*, 21(1), 128-138.
    https://doi.org/10.1097/EDE.0b013e3181c30fb2
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)

    # Hosmer-Lemeshow test
    order = np.argsort(y_pred)
    groups = np.array_split(order, n_groups)
    hl_stat = 0.0
    for grp in groups:
        obs = y_true[grp].sum()
        exp = y_pred[grp].sum()
        n_g = len(grp)
        avg_p = y_pred[grp].mean()
        if avg_p > 0 and avg_p < 1:
            hl_stat += (obs - exp) ** 2 / (n_g * avg_p * (1 - avg_p))
    hl_df = n_groups - 2
    hl_p = float(sp_stats.chi2.sf(hl_stat, hl_df))

    # Calibration slope and intercept (logistic recalibration)
    from sklearn.linear_model import LogisticRegression

    logit_pred = np.log(np.clip(y_pred, 1e-10, 1 - 1e-10) / (1 - np.clip(y_pred, 1e-10, 1 - 1e-10)))
    lr = LogisticRegression(penalty=None, max_iter=1000)
    lr.fit(logit_pred.reshape(-1, 1), y_true)
    cal_slope = float(lr.coef_[0, 0])
    cal_intercept = float(lr.intercept_[0])

    # Brier score
    brier = float(np.mean((y_pred - y_true) ** 2))
    prevalence = y_true.mean()
    brier_max = prevalence * (1 - prevalence)
    scaled_brier = 1 - brier / brier_max if brier_max > 0 else 0.0

    # Calibration-in-the-large
    citl = float(np.mean(y_pred) - np.mean(y_true))

    return CalibrationResult(
        hosmer_lemeshow_stat=float(hl_stat),
        hosmer_lemeshow_p=hl_p,
        calibration_slope=cal_slope,
        calibration_intercept=cal_intercept,
        brier_score=brier,
        scaled_brier=scaled_brier,
        calibration_in_the_large=citl,
    )


# ===========================================================================
# Discrimination assessment
# ===========================================================================


@dataclass
class DiscriminationResult:
    """Discrimination assessment metrics."""

    auroc: float
    auroc_ci_lower: float
    auroc_ci_upper: float
    c_statistic: float
    somers_d: float
    discrimination_slope: float
    nri: float | None = None
    idi: float | None = None


def assess_discrimination(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    y_pred_ref: np.ndarray | pd.Series | None = None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> DiscriminationResult:
    """
    Comprehensive discrimination assessment for a binary classifier.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted probabilities from the model being evaluated.
    y_pred_ref : array-like or None
        Predicted probabilities from a reference model.  If provided,
        NRI and IDI are computed comparing *y_pred* to *y_pred_ref*.
    n_bootstrap : int
        Number of bootstrap replicates for the AUC confidence interval
        (DeLong approximation is used when possible; bootstrap is fallback).
    confidence : float
    random_state : int

    Returns
    -------
    DiscriminationResult

    References
    ----------
    DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988).
    Comparing the areas under two or more correlated receiver operating
    characteristic curves. *Biometrics*, 44(3), 837-845.
    https://doi.org/10.2307/2531595

    Pencina, M. J., et al. (2008). Evaluating the added predictive
    ability of a new marker. *Statistics in Medicine*, 27(2), 157-172.
    https://doi.org/10.1002/sim.2929
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)

    auroc = roc_auc_score(y_true, y_pred)

    # Bootstrap CI for AUC
    rng = np.random.default_rng(random_state)
    boot_aucs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    boot_aucs = np.array(boot_aucs)
    alpha = (1 - confidence) / 2
    ci_lo = float(np.percentile(boot_aucs, 100 * alpha))
    ci_hi = float(np.percentile(boot_aucs, 100 * (1 - alpha)))

    # C-statistic = AUC for binary outcomes
    c_stat = auroc

    # Somers' D = 2 * (C - 0.5)
    somers_d = 2 * (c_stat - 0.5)

    # Discrimination slope = mean predicted prob (events) - mean (non-events)
    events = y_true == 1
    disc_slope = float(y_pred[events].mean() - y_pred[~events].mean())

    # NRI and IDI
    nri_val = None
    idi_val = None
    if y_pred_ref is not None:
        y_pred_ref = np.asarray(y_pred_ref, dtype=float)
        # Category-free NRI
        # Events: improvement = new pred is higher
        event_up = (y_pred[events] > y_pred_ref[events]).sum() - (y_pred[events] < y_pred_ref[events]).sum()
        nri_events = event_up / events.sum() if events.sum() > 0 else 0
        # Non-events: improvement = new pred is lower
        nonevent_up = (y_pred[~events] < y_pred_ref[~events]).sum() - (y_pred[~events] > y_pred_ref[~events]).sum()
        nri_nonevents = nonevent_up / (~events).sum() if (~events).sum() > 0 else 0
        nri_val = float(nri_events + nri_nonevents)

        # IDI = change in discrimination slope
        ref_disc = float(y_pred_ref[events].mean() - y_pred_ref[~events].mean())
        idi_val = disc_slope - ref_disc

    return DiscriminationResult(
        auroc=auroc,
        auroc_ci_lower=ci_lo,
        auroc_ci_upper=ci_hi,
        c_statistic=c_stat,
        somers_d=somers_d,
        discrimination_slope=disc_slope,
        nri=nri_val,
        idi=idi_val,
    )


# ===========================================================================
# Decision curve analysis
# ===========================================================================


@dataclass
class DecisionCurveResult:
    """Decision curve analysis result."""

    thresholds: np.ndarray
    net_benefit: np.ndarray
    net_benefit_all: np.ndarray
    net_benefit_none: np.ndarray


def decision_curve_analysis(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    thresholds: np.ndarray | None = None,
) -> DecisionCurveResult:
    """
    Decision curve analysis for clinical prediction models.

    Computes net benefit across a range of probability thresholds.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted probabilities.
    thresholds : array-like or None
        Probability thresholds to evaluate.  Defaults to
        ``np.arange(0.01, 0.99, 0.01)``.

    Returns
    -------
    DecisionCurveResult

    References
    ----------
    Vickers, A. J., & Elkin, E. B. (2006). Decision curve analysis:
    a novel method for evaluating prediction models. *Medical Decision
    Making*, 26(6), 565-574.
    https://doi.org/10.1177/0272989X06295361
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)

    if thresholds is None:
        thresholds = np.arange(0.01, 0.99, 0.01)
    thresholds = np.asarray(thresholds, dtype=float)

    prevalence = y_true.mean()
    net_benefit = np.zeros(len(thresholds))
    net_benefit_all = np.zeros(len(thresholds))
    net_benefit_none = np.zeros(len(thresholds))

    for i, t in enumerate(thresholds):
        # Model
        predicted_pos = y_pred >= t
        tp = np.sum(predicted_pos & (y_true == 1))
        fp = np.sum(predicted_pos & (y_true == 0))
        net_benefit[i] = (tp / n) - (fp / n) * (t / (1 - t))

        # Treat all
        net_benefit_all[i] = prevalence - (1 - prevalence) * (t / (1 - t))

        # Treat none
        net_benefit_none[i] = 0.0

    return DecisionCurveResult(
        thresholds=thresholds,
        net_benefit=net_benefit,
        net_benefit_all=net_benefit_all,
        net_benefit_none=net_benefit_none,
    )


# ===========================================================================
# Overfitting detection
# ===========================================================================


@dataclass
class OverfitResult:
    """Overfitting assessment result."""

    apparent_performance: float
    optimism: float
    corrected_performance: float
    shrinkage_factor: float
    recommendation: str


def detect_overfitting(
    estimator: BaseEstimator,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    scoring: str = "roc_auc",
    n_bootstrap: int = 200,
    random_state: int = 42,
) -> OverfitResult:
    """
    Detect overfitting via bootstrap optimism correction.

    Estimates the optimism (the gap between apparent and honest performance)
    and computes a shrinkage factor.

    Parameters
    ----------
    estimator : sklearn estimator
    X : array-like
    y : array-like
    scoring : str
    n_bootstrap : int
    random_state : int

    Returns
    -------
    OverfitResult

    References
    ----------
    Steyerberg, E. W. (2019). *Clinical Prediction Models* (2nd ed.).
    Springer. https://doi.org/10.1007/978-3-030-16399-0
    """
    from sklearn.metrics import get_scorer

    scorer = get_scorer(scoring)
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    n = len(y_arr)
    rng = np.random.default_rng(random_state)

    # Apparent performance
    est_app = clone(estimator).fit(X_arr, y_arr)
    apparent = scorer(est_app, X_arr, y_arr)

    optimism_vals = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_arr[idx])) < 2:
            continue
        est_b = clone(estimator).fit(X_arr[idx], y_arr[idx])
        perf_boot = scorer(est_b, X_arr[idx], y_arr[idx])
        perf_orig = scorer(est_b, X_arr, y_arr)
        optimism_vals.append(perf_boot - perf_orig)

    optimism = float(np.mean(optimism_vals))
    corrected = apparent - optimism
    shrinkage = corrected / apparent if apparent > 0 else 1.0

    if optimism > 0.05:
        rec = "Substantial overfitting detected. Consider regularisation or simpler model."
    elif optimism > 0.02:
        rec = "Moderate optimism. Bootstrap-corrected estimates recommended."
    else:
        rec = "Minimal overfitting. Model performance appears stable."

    return OverfitResult(
        apparent_performance=apparent,
        optimism=optimism,
        corrected_performance=corrected,
        shrinkage_factor=shrinkage,
        recommendation=rec,
    )


# ===========================================================================
# Temporal validation
# ===========================================================================


@dataclass
class TemporalValidationResult:
    """Temporal (time-split) validation result."""

    train_score: float
    test_score: float
    degradation: float
    split_date: str
    train_n: int
    test_n: int


def temporal_validate(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    date_col: str,
    *,
    split_date: str | pd.Timestamp | None = None,
    split_quantile: float = 0.7,
    scoring: str = "roc_auc",
) -> TemporalValidationResult:
    """
    Temporal validation: train on earlier data, test on later data.

    Parameters
    ----------
    estimator : sklearn estimator
    X : DataFrame
        Feature matrix (must include or align with *date_col*).
    y : Series
        Target variable.
    date_col : str
        Date column name in *X*.
    split_date : str, Timestamp, or None
        Date to split on.  If ``None``, uses *split_quantile*.
    split_quantile : float
        Quantile of dates for the split (default 0.7 = first 70% for
        training).
    scoring : str

    Returns
    -------
    TemporalValidationResult
    """
    from sklearn.metrics import get_scorer

    scorer = get_scorer(scoring)

    dates = pd.to_datetime(X[date_col])
    if split_date is None:
        split_date = dates.quantile(split_quantile)
    else:
        split_date = pd.Timestamp(split_date)

    train_mask = dates <= split_date
    test_mask = dates > split_date

    feature_cols = [c for c in X.columns if c != date_col]
    X_train = X.loc[train_mask, feature_cols].values
    X_test = X.loc[test_mask, feature_cols].values
    y_train = np.asarray(y[train_mask])
    y_test = np.asarray(y[test_mask])

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError("Both train and test sets must have at least 2 classes")

    est = clone(estimator).fit(X_train, y_train)
    train_score = scorer(est, X_train, y_train)
    test_score = scorer(est, X_test, y_test)
    degradation = train_score - test_score

    return TemporalValidationResult(
        train_score=float(train_score),
        test_score=float(test_score),
        degradation=float(degradation),
        split_date=str(split_date.date()),
        train_n=int(train_mask.sum()),
        test_n=int(test_mask.sum()),
    )


# ===========================================================================
# External validation framework
# ===========================================================================


@dataclass
class ExternalValidationResult:
    """External validation result."""

    discrimination: DiscriminationResult
    calibration: CalibrationResult
    n_external: int
    domain_shift: dict[str, float]


def external_validate(
    model: BaseEstimator,
    X_external: np.ndarray | pd.DataFrame,
    y_external: np.ndarray | pd.Series,
    *,
    X_development: np.ndarray | pd.DataFrame | None = None,
) -> ExternalValidationResult:
    """
    External validation of a fitted model on new data.

    Parameters
    ----------
    model : fitted sklearn estimator
        Must have a ``predict_proba`` method.
    X_external : array-like
        External data features.
    y_external : array-like
        External data outcome.
    X_development : array-like or None
        Original development data features (for domain-shift diagnostics).

    Returns
    -------
    ExternalValidationResult
    """
    X_ext = np.asarray(X_external)
    y_ext = np.asarray(y_external, dtype=int)

    y_pred = model.predict_proba(X_ext)[:, 1]

    disc = assess_discrimination(y_ext, y_pred)
    cal = assess_calibration(y_ext, y_pred)

    # Domain shift: compare feature distributions
    domain_shift: dict[str, float] = {}
    if X_development is not None:
        X_dev = np.asarray(X_development)
        for j in range(X_ext.shape[1]):
            stat, p = sp_stats.ks_2samp(X_dev[:, j], X_ext[:, j])
            domain_shift[f"feature_{j}"] = float(p)

    return ExternalValidationResult(
        discrimination=disc,
        calibration=cal,
        n_external=len(y_ext),
        domain_shift=domain_shift,
    )


# ===========================================================================
# Reproducibility manifest
# ===========================================================================


@dataclass
class ReproducibilityManifest:
    """Manifest capturing all information needed to reproduce an analysis."""

    python_version: str
    package_versions: dict[str, str]
    random_seeds: dict[str, int]
    data_checksum: str
    parameters: dict[str, Any]
    timestamp: str


def create_reproducibility_manifest(
    data: pd.DataFrame,
    *,
    parameters: dict[str, Any] | None = None,
    seeds: dict[str, int] | None = None,
) -> ReproducibilityManifest:
    """
    Create a reproducibility manifest for an analysis.

    Parameters
    ----------
    data : DataFrame
        The analysis data (used to compute a checksum).
    parameters : dict or None
        Key analysis parameters.
    seeds : dict or None
        Random seeds used.

    Returns
    -------
    ReproducibilityManifest
    """
    import datetime
    import sys

    # Data checksum
    buf = data.to_csv(index=False).encode("utf-8")
    checksum = hashlib.sha256(buf).hexdigest()

    # Package versions
    versions: dict[str, str] = {}
    for pkg_name in ["pandas", "numpy", "scipy", "sklearn", "statsmodels", "matplotlib"]:
        try:
            mod = __import__(pkg_name)
            versions[pkg_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg_name] = "not installed"

    return ReproducibilityManifest(
        python_version=sys.version,
        package_versions=versions,
        random_seeds=seeds or {},
        data_checksum=checksum,
        parameters=parameters or {},
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
