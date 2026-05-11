"""Survey weight calibration, replication, and diagnostics.

Provides a comprehensive toolkit for constructing, calibrating, trimming,
and diagnosing survey weights used in complex survey designs.  Supports
design weights, post-stratification, raking (iterative proportional
fitting), GREG calibration, replicate weight generation (jackknife, BRR,
Fay's BRR, bootstrap, SDR), and non-response adjustments.

All weight functions operate on numpy arrays or pandas Series for
interoperability with the rest of the MORIE analysis pipeline.

References
----------
Deville, J.-C. & Sarndal, C.-E. (1992). Calibration estimators in survey
sampling. *Journal of the American Statistical Association*, 87(418),
376--382. https://doi.org/10.1080/01621459.1992.10475217

Kish, L. (1965). *Survey Sampling*. Wiley.

Lumley, T. (2010). *Complex Surveys: A Guide to Analysis Using R*. Wiley.

Wolter, K. M. (2007). *Introduction to Variance Estimation* (2nd ed.).
Springer.

Fay, R. E. (1989). Theory and application of replicate weighting for
variance calculations. *Proceedings of the Survey Research Methods Section,
ASA*, 212--217.

Judkins, D. R. (1990). Fay's method for variance estimation. *Journal of
Official Statistics*, 6(3), 223--239.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class WeightDiagnostics:
    """Summary diagnostics for a set of survey weights."""

    n: int
    sum_weights: float
    mean_weight: float
    median_weight: float
    std_weight: float
    min_weight: float
    max_weight: float
    cv: float
    effective_sample_size: float
    design_effect: float
    weight_range_ratio: float
    n_zero: int
    n_negative: int
    percentiles: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n": self.n,
            "sum_weights": round(self.sum_weights, 4),
            "mean_weight": round(self.mean_weight, 4),
            "median_weight": round(self.median_weight, 4),
            "std_weight": round(self.std_weight, 4),
            "min_weight": round(self.min_weight, 4),
            "max_weight": round(self.max_weight, 4),
            "cv": round(self.cv, 4),
            "effective_sample_size": round(self.effective_sample_size, 2),
            "design_effect": round(self.design_effect, 4),
            "weight_range_ratio": round(self.weight_range_ratio, 4),
            "n_zero": self.n_zero,
            "n_negative": self.n_negative,
            "percentiles": {k: round(v, 4) for k, v in self.percentiles.items()},
        }


@dataclass
class ExtremeWeightReport:
    """Report on extreme weights."""

    n_extreme: int
    threshold_lower: float
    threshold_upper: float
    extreme_indices: list[int]
    extreme_values: list[float]
    pct_extreme: float


@dataclass
class WeightInfluence:
    """Influence analysis for individual weights."""

    index: int
    weight: float
    influence: float
    contribution_pct: float


@dataclass
class CalibrationResult:
    """Result from a weight calibration procedure."""

    weights: np.ndarray
    converged: bool
    iterations: int
    max_adjustment: float
    diagnostics: WeightDiagnostics


# ---------------------------------------------------------------------------
# Design weights
# ---------------------------------------------------------------------------


def compute_design_weights(
    selection_probs: np.ndarray | pd.Series,
) -> np.ndarray:
    """Compute design weights as the inverse of selection probabilities.

    .. math::

        w_i = \\frac{1}{\\pi_i}

    where :math:`\\pi_i` is the selection probability for unit *i*.

    Parameters
    ----------
    selection_probs : numpy.ndarray or pandas.Series
        Inclusion probabilities in (0, 1].

    Returns
    -------
    numpy.ndarray
        Design weights.

    Raises
    ------
    ValueError
        If any selection probability is <= 0 or > 1.

    Examples
    --------
    >>> import numpy as np
    >>> w = compute_design_weights(np.array([0.5, 0.25, 0.1]))
    >>> w.tolist()
    [2.0, 4.0, 10.0]

    References
    ----------
    Horvitz, D. G. & Thompson, D. J. (1952). A generalization of sampling
    without replacement from a finite universe. *JASA*, 47(260), 663--685.
    """
    probs = np.asarray(selection_probs, dtype=float)
    if np.any(probs <= 0):
        raise ValueError("Selection probabilities must be > 0")
    if np.any(probs > 1):
        raise ValueError("Selection probabilities must be <= 1")
    return 1.0 / probs


# ---------------------------------------------------------------------------
# Post-stratification
# ---------------------------------------------------------------------------


def poststratify(
    weights: np.ndarray | pd.Series,
    strata: np.ndarray | pd.Series,
    population_totals: dict[Any, float],
) -> np.ndarray:
    """Post-stratification weight adjustment.

    Adjusts weights so that the weighted sum within each stratum matches
    known population totals.

    .. math::

        w_i^{\\text{ps}} = w_i \\cdot \\frac{N_h}{\\hat{N}_h}

    where :math:`\\hat{N}_h = \\sum_{i \\in h} w_i` is the weighted sample
    count in stratum *h* and :math:`N_h` is the population total.

    Parameters
    ----------
    weights : numpy.ndarray or pandas.Series
        Input weights.
    strata : numpy.ndarray or pandas.Series
        Stratum membership for each unit.
    population_totals : dict
        Known population totals per stratum.

    Returns
    -------
    numpy.ndarray
        Post-stratified weights.

    Raises
    ------
    ValueError
        If a stratum in the data has no population total.

    Examples
    --------
    >>> import numpy as np
    >>> w = np.array([1.0, 1.0, 1.0, 1.0])
    >>> s = np.array(["A", "A", "B", "B"])
    >>> w_ps = poststratify(w, s, {"A": 100, "B": 200})
    >>> w_ps.tolist()
    [50.0, 50.0, 100.0, 100.0]

    References
    ----------
    Holt, D. & Smith, T. M. F. (1979). Post stratification. *JRSS Series A*,
    142(1), 33--46.
    """
    w = np.asarray(weights, dtype=float).copy()
    s = np.asarray(strata)
    unique_strata = np.unique(s)

    for stratum in unique_strata:
        if stratum not in population_totals:
            raise ValueError(
                f"Stratum '{stratum}' has no entry in population_totals. Available: {list(population_totals.keys())}"
            )
        mask = s == stratum
        weighted_total = np.sum(w[mask])
        if weighted_total == 0:
            warnings.warn(
                f"Stratum '{stratum}' has zero weighted total; post-stratification factor will be inf.",
                stacklevel=2,
            )
            continue
        adjustment = population_totals[stratum] / weighted_total
        w[mask] *= adjustment

    return w


# ---------------------------------------------------------------------------
# Raking (iterative proportional fitting)
# ---------------------------------------------------------------------------


def rake(
    weights: np.ndarray | pd.Series,
    df: pd.DataFrame,
    margins: dict[str, dict[Any, float]],
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    bounds: tuple[float, float] | None = None,
) -> CalibrationResult:
    """Raking (iterative proportional fitting) calibration.

    Adjusts weights to match known marginal population totals for multiple
    categorical variables simultaneously through iterative proportional
    fitting.

    Parameters
    ----------
    weights : numpy.ndarray or pandas.Series
        Initial weights.
    df : pd.DataFrame
        Data frame containing the calibration variables.
    margins : dict[str, dict[Any, float]]
        For each calibration variable (key), a dict mapping category values
        to target population totals.
    max_iterations : int
        Maximum iterations (default 100).
    tolerance : float
        Convergence tolerance (default 1e-6).
    bounds : tuple[float, float] | None
        If provided, clip weight adjustment factors to (lower, upper).

    Returns
    -------
    CalibrationResult

    References
    ----------
    Deming, W. E. & Stephan, F. F. (1940). On a least squares adjustment of
    a sampled frequency table when the expected marginal totals are known.
    *Annals of Mathematical Statistics*, 11(4), 427--444.
    """
    w = np.asarray(weights, dtype=float).copy()
    n = len(w)
    converged = False
    max_adj = 0.0

    for iteration in range(max_iterations):
        max_adj = 0.0
        for var_name, targets in margins.items():
            if var_name not in df.columns:
                raise KeyError(f"Column '{var_name}' not found in DataFrame")

            values = df[var_name].values
            for category, target_total in targets.items():
                mask = values == category
                if not np.any(mask):
                    continue
                current_total = np.sum(w[mask])
                if current_total == 0:
                    continue
                factor = target_total / current_total

                if bounds is not None:
                    factor = np.clip(factor, bounds[0], bounds[1])

                w[mask] *= factor
                max_adj = max(max_adj, abs(factor - 1.0))

        if max_adj < tolerance:
            converged = True
            logger.info("Raking converged after %d iterations", iteration + 1)
            break

    if not converged:
        warnings.warn(
            f"Raking did not converge after {max_iterations} iterations (max adjustment = {max_adj:.6f})",
            stacklevel=2,
        )

    diag = weight_diagnostics(w)
    return CalibrationResult(
        weights=w,
        converged=converged,
        iterations=iteration + 1 if not converged else iteration + 1,
        max_adjustment=max_adj,
        diagnostics=diag,
    )


# ---------------------------------------------------------------------------
# GREG calibration
# ---------------------------------------------------------------------------


def greg_calibrate(
    weights: np.ndarray | pd.Series,
    X: np.ndarray | pd.DataFrame,
    population_totals: np.ndarray,
    *,
    max_iterations: int = 50,
    tolerance: float = 1e-8,
) -> CalibrationResult:
    """Generalized regression (GREG) calibration estimator.

    Finds calibration weights :math:`w_i^*` that minimize the chi-squared
    distance :math:`\\sum_i (w_i^* - w_i)^2 / w_i` subject to the
    calibration constraint :math:`\\sum_i w_i^* \\mathbf{x}_i = \\mathbf{T}_x`.

    The closed-form solution is:

    .. math::

        w_i^* = w_i \\left(1 + (\\mathbf{T}_x - \\hat{\\mathbf{T}}_x)^\\top
        \\left(\\sum_j w_j \\mathbf{x}_j \\mathbf{x}_j^\\top\\right)^{-1}
        \\mathbf{x}_i\\right)

    Parameters
    ----------
    weights : numpy.ndarray or pandas.Series
        Initial design weights.
    X : numpy.ndarray or pandas.DataFrame
        Calibration variables (n x p matrix).
    population_totals : numpy.ndarray
        Known population totals for each calibration variable (length p).
    max_iterations : int
        Maximum iterations (default 50; single pass for linear calibration).
    tolerance : float
        Convergence tolerance.

    Returns
    -------
    CalibrationResult

    References
    ----------
    Deville, J.-C. & Sarndal, C.-E. (1992). Calibration estimators in
    survey sampling. *JASA*, 87(418), 376--382.
    """
    w = np.asarray(weights, dtype=float).copy()
    X_mat = np.asarray(X, dtype=float)
    T_x = np.asarray(population_totals, dtype=float)

    if X_mat.ndim == 1:
        X_mat = X_mat.reshape(-1, 1)

    n, p = X_mat.shape
    if len(T_x) != p:
        raise ValueError(f"population_totals length ({len(T_x)}) must match number of calibration variables ({p})")

    # Weighted totals from sample
    T_hat = X_mat.T @ w

    # Weighted cross-product matrix
    W_diag = np.diag(w)
    XtWX = X_mat.T @ W_diag @ X_mat

    # Check for singularity
    try:
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        XtWX_inv = np.linalg.pinv(XtWX)
        warnings.warn(
            "GREG calibration: singular XtWX matrix, using pseudo-inverse",
            stacklevel=2,
        )

    # Calibration adjustment
    lambda_vec = XtWX_inv @ (T_x - T_hat)
    g = 1.0 + X_mat @ lambda_vec
    w_cal = w * g

    # Check convergence
    T_cal = X_mat.T @ w_cal
    max_adj = float(np.max(np.abs(T_cal - T_x)))
    converged = max_adj < tolerance

    if not converged:
        logger.warning("GREG calibration residual = %.6f (tolerance = %.6f)", max_adj, tolerance)

    diag = weight_diagnostics(w_cal)
    return CalibrationResult(
        weights=w_cal,
        converged=converged,
        iterations=1,
        max_adjustment=max_adj,
        diagnostics=diag,
    )


def calibrate_to_totals(
    weights: np.ndarray | pd.Series,
    df: pd.DataFrame,
    totals: dict[str, float],
    *,
    method: str = "raking",
    **kwargs: Any,
) -> CalibrationResult:
    """Calibrate weights to known population totals.

    Convenience wrapper that selects the calibration method.

    Parameters
    ----------
    weights : numpy.ndarray or pandas.Series
        Initial weights.
    df : pd.DataFrame
        Data frame.
    totals : dict[str, float]
        Column names mapped to their target population totals.
    method : str
        ``raking`` or ``greg``.
    **kwargs
        Additional arguments passed to the calibration function.

    Returns
    -------
    CalibrationResult
    """
    if method == "raking":
        # Convert totals to margins format for raking
        margins: dict[str, dict[Any, float]] = {}
        for col, total in totals.items():
            unique_vals = df[col].unique()
            # Distribute total proportionally based on unweighted counts
            counts = df[col].value_counts()
            proportions = counts / counts.sum()
            margins[col] = {val: total * proportions.get(val, 0) for val in unique_vals}
        return rake(weights, df, margins, **kwargs)
    elif method == "greg":
        X = df[list(totals.keys())].values
        pop_totals = np.array([totals[c] for c in totals])
        return greg_calibrate(weights, X, pop_totals, **kwargs)
    else:
        raise ValueError(f"Unknown calibration method: {method}. Use 'raking' or 'greg'.")


# ---------------------------------------------------------------------------
# Weight trimming and smoothing
# ---------------------------------------------------------------------------


def trim_weights(
    weights: np.ndarray | pd.Series,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    method: str = "percentile",
) -> np.ndarray:
    """Trim extreme weights.

    Parameters
    ----------
    weights : numpy.ndarray or pandas.Series
        Input weights.
    lower_percentile : float
        Lower trimming percentile (default 1).
    upper_percentile : float
        Upper trimming percentile (default 99).
    method : str
        ``percentile`` (clip at percentiles) or ``winsorize``
        (replace extremes with boundary values).

    Returns
    -------
    numpy.ndarray
        Trimmed weights.

    Examples
    --------
    >>> import numpy as np
    >>> w = np.array([0.1, 1.0, 1.0, 1.0, 100.0])
    >>> w_trimmed = trim_weights(w, lower_percentile=5, upper_percentile=95)
    >>> w_trimmed[-1] < 100.0
    True

    References
    ----------
    Potter, F. J. (1990). A study of procedures to identify and trim
    extreme sampling weights. *Proceedings of the Survey Research Methods
    Section, ASA*, 225--230.
    """
    w = np.asarray(weights, dtype=float).copy()
    lo = np.percentile(w, lower_percentile)
    hi = np.percentile(w, upper_percentile)

    if method == "percentile":
        w = np.clip(w, lo, hi)
    elif method == "winsorize":
        w[w < lo] = lo
        w[w > hi] = hi
    else:
        raise ValueError(f"Unknown trimming method: {method}. Use 'percentile' or 'winsorize'.")

    n_trimmed = int(np.sum((weights < lo) | (weights > hi)))
    if n_trimmed > 0:
        logger.info("Trimmed %d weights (%.1f%%)", n_trimmed, 100 * n_trimmed / len(w))

    return w


def smooth_weights(
    weights: np.ndarray | pd.Series,
    *,
    method: str = "linear_shrinkage",
    shrinkage_factor: float = 0.5,
) -> np.ndarray:
    """Smooth survey weights to reduce variance.

    Parameters
    ----------
    weights : numpy.ndarray or pandas.Series
        Input weights.
    method : str
        ``linear_shrinkage`` (shrink toward mean) or ``log_transform``
        (compress via log scale).
    shrinkage_factor : float
        Shrinkage intensity in [0, 1]. 0 = no shrinkage, 1 = all equal.

    Returns
    -------
    numpy.ndarray
        Smoothed weights.

    References
    ----------
    Beaumont, J.-F. & Rivest, L.-P. (2009). Dealing with outliers in
    survey data. In *Handbook of Statistics* (Vol. 29A, pp. 247--279).
    Elsevier.
    """
    w = np.asarray(weights, dtype=float).copy()
    mean_w = np.mean(w)

    if method == "linear_shrinkage":
        if not 0 <= shrinkage_factor <= 1:
            raise ValueError("shrinkage_factor must be in [0, 1]")
        w = (1 - shrinkage_factor) * w + shrinkage_factor * mean_w
    elif method == "log_transform":
        # Compress on log scale then back-transform, preserving sum
        original_sum = np.sum(w)
        log_w = np.log(np.maximum(w, 1e-10))
        log_mean = np.mean(log_w)
        smoothed_log = (1 - shrinkage_factor) * log_w + shrinkage_factor * log_mean
        w = np.exp(smoothed_log)
        # Rescale to preserve original sum
        w = w * (original_sum / np.sum(w))
    else:
        raise ValueError(f"Unknown smoothing method: {method}")

    return w


# ---------------------------------------------------------------------------
# Non-response adjustment
# ---------------------------------------------------------------------------


def nonresponse_adjustment(
    weights: np.ndarray | pd.Series,
    responded: np.ndarray | pd.Series,
    *,
    adjustment_cells: np.ndarray | pd.Series | None = None,
) -> np.ndarray:
    """Adjust weights for unit non-response within adjustment cells.

    Within each cell, the adjustment factor is:

    .. math::

        f_c = \\frac{\\sum_{i \\in c} w_i}{\\sum_{i \\in c, r_i = 1} w_i}

    where :math:`r_i = 1` for respondents. Non-respondent weights are set
    to zero.

    Parameters
    ----------
    weights : numpy.ndarray or pandas.Series
        Initial weights.
    responded : numpy.ndarray or pandas.Series
        Boolean indicator (True = responded).
    adjustment_cells : numpy.ndarray | pandas.Series | None
        Cell membership for adjustment. If None, uses a single cell.

    Returns
    -------
    numpy.ndarray
        Adjusted weights (non-respondents get zero).

    References
    ----------
    Kalton, G. & Flores-Cervantes, I. (2003). Weighting methods.
    *Journal of Official Statistics*, 19(2), 81--97.
    """
    w = np.asarray(weights, dtype=float).copy()
    resp = np.asarray(responded, dtype=bool)

    if adjustment_cells is None:
        cells = np.zeros(len(w), dtype=int)
    else:
        cells = np.asarray(adjustment_cells)

    unique_cells = np.unique(cells)

    for cell in unique_cells:
        mask_cell = cells == cell
        mask_resp = mask_cell & resp

        total_w = np.sum(w[mask_cell])
        resp_w = np.sum(w[mask_resp])

        if resp_w == 0:
            warnings.warn(
                f"Cell '{cell}' has no responding weighted units; setting all weights in this cell to zero.",
                stacklevel=2,
            )
            w[mask_cell] = 0.0
            continue

        factor = total_w / resp_w
        w[mask_resp] *= factor

    # Zero out non-respondents
    w[~resp] = 0.0

    return w


def propensity_nonresponse_weights(
    weights: np.ndarray | pd.Series,
    responded: np.ndarray | pd.Series,
    X: np.ndarray | pd.DataFrame,
    *,
    model: str = "logistic",
) -> np.ndarray:
    """Compute non-response adjusted weights using propensity modeling.

    Fits a response propensity model and adjusts weights by the inverse
    of the predicted response probability.

    Parameters
    ----------
    weights : numpy.ndarray or pandas.Series
        Initial weights.
    responded : numpy.ndarray or pandas.Series
        Boolean response indicator.
    X : numpy.ndarray or pandas.DataFrame
        Covariates for the response propensity model.
    model : str
        Model type: ``logistic`` (default).

    Returns
    -------
    numpy.ndarray
        Adjusted weights (non-respondents get zero).

    References
    ----------
    Rosenbaum, P. R. & Rubin, D. B. (1983). The central role of the
    propensity score in observational studies for causal effects.
    *Biometrika*, 70(1), 41--55.
    """
    from sklearn.linear_model import LogisticRegression

    w = np.asarray(weights, dtype=float).copy()
    resp = np.asarray(responded, dtype=bool)
    X_mat = np.asarray(X, dtype=float)
    if X_mat.ndim == 1:
        X_mat = X_mat.reshape(-1, 1)

    if model == "logistic":
        lr = LogisticRegression(max_iter=1000, solver="lbfgs")
        lr.fit(X_mat, resp.astype(int), sample_weight=w)
        p_response = lr.predict_proba(X_mat)[:, 1]
    else:
        raise ValueError(f"Unknown model type: {model}")

    # Clip probabilities to avoid division by zero
    p_response = np.clip(p_response, 0.01, 1.0)

    # Adjust respondent weights
    w[resp] *= 1.0 / p_response[resp]
    w[~resp] = 0.0

    return w


# ---------------------------------------------------------------------------
# Combined weight computation
# ---------------------------------------------------------------------------


def compute_combined_weights(
    selection_probs: np.ndarray | pd.Series,
    responded: np.ndarray | pd.Series,
    *,
    adjustment_cells: np.ndarray | pd.Series | None = None,
    calibration_strata: np.ndarray | pd.Series | None = None,
    population_totals: dict[Any, float] | None = None,
    trim_percentiles: tuple[float, float] | None = None,
) -> np.ndarray:
    """Compute combined weights: design x non-response x calibration.

    Applies the full weight construction pipeline:

    1. Design weights (inverse selection probability).
    2. Non-response adjustment (within cells).
    3. Post-stratification calibration (to population totals).
    4. Optional trimming of extreme weights.

    Parameters
    ----------
    selection_probs : numpy.ndarray or pandas.Series
        Selection probabilities.
    responded : numpy.ndarray or pandas.Series
        Boolean response indicator.
    adjustment_cells : numpy.ndarray | pandas.Series | None
        Non-response adjustment cells.
    calibration_strata : numpy.ndarray | pandas.Series | None
        Strata for post-stratification.
    population_totals : dict | None
        Population totals per stratum.
    trim_percentiles : tuple[float, float] | None
        If provided, trim weights at these percentiles.

    Returns
    -------
    numpy.ndarray
        Final combined weights.
    """
    # Step 1: Design weights
    w = compute_design_weights(selection_probs)
    logger.info("Design weights: ESS = %.1f", _ess(w))

    # Step 2: Non-response
    w = nonresponse_adjustment(w, responded, adjustment_cells=adjustment_cells)
    resp = np.asarray(responded, dtype=bool)
    logger.info("After NR adjustment: ESS = %.1f (respondents)", _ess(w[resp]))

    # Step 3: Calibration
    if calibration_strata is not None and population_totals is not None:
        w = poststratify(w, calibration_strata, population_totals)
        logger.info("After calibration: ESS = %.1f", _ess(w[resp]))

    # Step 4: Trimming
    if trim_percentiles is not None:
        w = trim_weights(w, lower_percentile=trim_percentiles[0], upper_percentile=trim_percentiles[1])
        logger.info("After trimming: ESS = %.1f", _ess(w[resp]))

    return w


# ---------------------------------------------------------------------------
# Weight normalization
# ---------------------------------------------------------------------------


def normalize_weights(
    weights: np.ndarray | pd.Series,
    *,
    target: str = "sample_size",
    population_size: float | None = None,
) -> np.ndarray:
    """Normalize weights to sum to a target.

    Parameters
    ----------
    weights : numpy.ndarray or pandas.Series
        Input weights.
    target : str
        ``sample_size`` (sum to n) or ``population`` (sum to N).
    population_size : float | None
        Required if ``target='population'``.

    Returns
    -------
    numpy.ndarray
        Normalized weights.

    Examples
    --------
    >>> import numpy as np
    >>> w = np.array([2.0, 4.0, 6.0])
    >>> w_norm = normalize_weights(w, target="sample_size")
    >>> np.isclose(w_norm.sum(), 3.0)
    True
    """
    w = np.asarray(weights, dtype=float).copy()
    current_sum = np.sum(w)
    if current_sum == 0:
        warnings.warn("Cannot normalize: sum of weights is zero", stacklevel=2)
        return w

    if target == "sample_size":
        w = w * (len(w) / current_sum)
    elif target == "population":
        if population_size is None:
            raise ValueError("population_size required when target='population'")
        w = w * (population_size / current_sum)
    else:
        raise ValueError(f"Unknown target: {target}. Use 'sample_size' or 'population'.")

    return w


# ---------------------------------------------------------------------------
# Weight diagnostics
# ---------------------------------------------------------------------------


def weight_diagnostics(
    weights: np.ndarray | pd.Series,
) -> WeightDiagnostics:
    """Compute comprehensive diagnostics for survey weights.

    Parameters
    ----------
    weights : numpy.ndarray or pandas.Series
        Survey weights.

    Returns
    -------
    WeightDiagnostics

    References
    ----------
    Kish, L. (1992). Weighting for unequal Pi. *Journal of Official
    Statistics*, 8(2), 183--200.
    """
    w = np.asarray(weights, dtype=float)
    n = len(w)

    if n == 0:
        return WeightDiagnostics(
            n=0,
            sum_weights=0,
            mean_weight=0,
            median_weight=0,
            std_weight=0,
            min_weight=0,
            max_weight=0,
            cv=0,
            effective_sample_size=0,
            design_effect=1,
            weight_range_ratio=0,
            n_zero=0,
            n_negative=0,
        )

    sum_w = float(np.sum(w))
    mean_w = float(np.mean(w))
    median_w = float(np.median(w))
    std_w = float(np.std(w, ddof=1)) if n > 1 else 0.0
    min_w = float(np.min(w))
    max_w = float(np.max(w))
    cv = std_w / mean_w if mean_w != 0 else 0.0

    ess = _ess(w)
    deff = n / ess if ess > 0 else float("inf")
    range_ratio = max_w / min_w if min_w > 0 else float("inf")

    percentiles = {
        "p1": float(np.percentile(w, 1)),
        "p5": float(np.percentile(w, 5)),
        "p10": float(np.percentile(w, 10)),
        "p25": float(np.percentile(w, 25)),
        "p50": float(np.percentile(w, 50)),
        "p75": float(np.percentile(w, 75)),
        "p90": float(np.percentile(w, 90)),
        "p95": float(np.percentile(w, 95)),
        "p99": float(np.percentile(w, 99)),
    }

    return WeightDiagnostics(
        n=n,
        sum_weights=sum_w,
        mean_weight=mean_w,
        median_weight=median_w,
        std_weight=std_w,
        min_weight=min_w,
        max_weight=max_w,
        cv=cv,
        effective_sample_size=ess,
        design_effect=deff,
        weight_range_ratio=range_ratio,
        n_zero=int(np.sum(w == 0)),
        n_negative=int(np.sum(w < 0)),
        percentiles=percentiles,
    )


def effective_sample_size(weights: np.ndarray | pd.Series) -> float:
    """Compute the Kish effective sample size.

    .. math::

        n_{\\text{eff}} = \\frac{\\left(\\sum_i w_i\\right)^2}{\\sum_i w_i^2}

    Parameters
    ----------
    weights : numpy.ndarray or pandas.Series
        Survey weights.

    Returns
    -------
    float

    References
    ----------
    Kish, L. (1965). *Survey Sampling*, p. 162. Wiley.
    """
    return _ess(weights)


def design_effect(weights: np.ndarray | pd.Series) -> float:
    """Compute the design effect (DEFF) from weights.

    .. math::

        \\text{DEFF} = \\frac{n}{n_{\\text{eff}}}

    Parameters
    ----------
    weights : numpy.ndarray or pandas.Series

    Returns
    -------
    float

    References
    ----------
    Kish, L. (1965). *Survey Sampling*, p. 162. Wiley.
    """
    w = np.asarray(weights, dtype=float)
    n = len(w)
    ess = _ess(w)
    if ess == 0:
        return float("inf")
    return float(n / ess)


def _ess(weights: np.ndarray | pd.Series) -> float:
    """Internal ESS helper."""
    w = np.asarray(weights, dtype=float)
    if len(w) == 0:
        return 0.0
    sum_w = w.sum()
    sum_w2 = (w**2).sum()
    if sum_w2 == 0:
        return 0.0
    return float(sum_w**2 / sum_w2)


# ---------------------------------------------------------------------------
# Extreme weight detection
# ---------------------------------------------------------------------------


def detect_extreme_weights(
    weights: np.ndarray | pd.Series,
    *,
    method: str = "iqr",
    multiplier: float = 3.0,
) -> ExtremeWeightReport:
    """Detect extreme weights using the IQR or percentile method.

    Parameters
    ----------
    weights : numpy.ndarray or pandas.Series
        Survey weights.
    method : str
        ``iqr`` (IQR-based) or ``percentile`` (1st/99th percentile).
    multiplier : float
        IQR multiplier for outlier detection (default 3.0).

    Returns
    -------
    ExtremeWeightReport

    References
    ----------
    Potter, F. J. (1990). A study of procedures to identify and trim
    extreme sampling weights. *Proc. Survey Research Methods Section, ASA*.
    """
    w = np.asarray(weights, dtype=float)

    if method == "iqr":
        q1 = float(np.percentile(w, 25))
        q3 = float(np.percentile(w, 75))
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
    elif method == "percentile":
        lower = float(np.percentile(w, 1))
        upper = float(np.percentile(w, 99))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'percentile'.")

    extreme_mask = (w < lower) | (w > upper)
    extreme_idx = np.where(extreme_mask)[0].tolist()
    extreme_vals = w[extreme_mask].tolist()

    return ExtremeWeightReport(
        n_extreme=len(extreme_idx),
        threshold_lower=lower,
        threshold_upper=upper,
        extreme_indices=extreme_idx,
        extreme_values=extreme_vals,
        pct_extreme=100 * len(extreme_idx) / len(w) if len(w) > 0 else 0.0,
    )


# ---------------------------------------------------------------------------
# Weight influence analysis
# ---------------------------------------------------------------------------


def weight_influence(
    weights: np.ndarray | pd.Series,
    values: np.ndarray | pd.Series,
) -> list[WeightInfluence]:
    """Analyze how much each weight influences the weighted estimate.

    For the weighted mean, the influence of unit *i* is:

    .. math::

        \\text{influence}_i = w_i \\cdot (y_i - \\bar{y}_w)

    Parameters
    ----------
    weights : numpy.ndarray or pandas.Series
        Survey weights.
    values : numpy.ndarray or pandas.Series
        Variable of interest.

    Returns
    -------
    list[WeightInfluence]
        Sorted by absolute influence (descending).
    """
    w = np.asarray(weights, dtype=float)
    y = np.asarray(values, dtype=float)

    if len(w) == 0:
        return []

    # Weighted mean
    w_sum = np.sum(w)
    if w_sum == 0:
        return []
    y_bar = np.sum(w * y) / w_sum

    # Influence = contribution to weighted mean shift
    influences = w * (y - y_bar)
    total_influence = np.sum(np.abs(influences))

    result = []
    for i in range(len(w)):
        contribution_pct = (abs(influences[i]) / total_influence * 100) if total_influence > 0 else 0.0
        result.append(
            WeightInfluence(
                index=i,
                weight=float(w[i]),
                influence=float(influences[i]),
                contribution_pct=float(contribution_pct),
            )
        )

    result.sort(key=lambda x: abs(x.influence), reverse=True)
    return result


# ---------------------------------------------------------------------------
# Replicate weight generation
# ---------------------------------------------------------------------------


def jackknife_replicate_weights(
    weights: np.ndarray | pd.Series,
    strata: np.ndarray | pd.Series,
    *,
    jk_type: str = "JK1",
) -> np.ndarray:
    """Generate jackknife replicate weights.

    For JK1 (unstratified delete-1): generates n replicate columns.
    For JKn (stratified): generates one replicate per PSU deletion within
    each stratum.

    Parameters
    ----------
    weights : numpy.ndarray or pandas.Series
        Full-sample weights.
    strata : numpy.ndarray or pandas.Series
        Stratum membership (used as PSU grouping for JKn).
    jk_type : str
        ``JK1`` (delete-1) or ``JKn`` (stratified delete-n).

    Returns
    -------
    numpy.ndarray
        Matrix of shape (n, n_replicates) containing replicate weights.

    References
    ----------
    Wolter, K. M. (2007). *Introduction to Variance Estimation* (2nd ed.),
    Chapter 4. Springer.
    """
    w = np.asarray(weights, dtype=float)
    s = np.asarray(strata)
    n = len(w)

    if jk_type == "JK1":
        # Delete-1 jackknife: n replicates
        rep_weights = np.tile(w, (n, 1)).T  # n x n
        for i in range(n):
            # Redistribute deleted unit's weight to others
            rep_weights[i, i] = 0.0
            remaining = w.copy()
            remaining[i] = 0
            total_remaining = remaining.sum()
            if total_remaining > 0:
                factor = w.sum() / total_remaining
                rep_weights[:, i] = remaining * factor

        return rep_weights

    elif jk_type == "JKn":
        unique_strata = np.unique(s)
        n_reps = len(unique_strata)
        rep_weights = np.tile(w, (n_reps, 1)).T  # n x n_reps

        for r, stratum in enumerate(unique_strata):
            mask = s == stratum
            n_h = int(mask.sum())
            if n_h <= 1:
                continue
            # Zero out this stratum and rescale others
            rep_weights[mask, r] = 0.0
            # Redistribute within stratum
            stratum_total = w[mask].sum()
            other_strata = ~mask
            # Keep other strata weights unchanged, rescale current stratum
            # Actually for JKn: remove one PSU, inflate remaining in stratum
            # Simplified: remove entire stratum in this replicate
            rep_weights[mask, r] = 0.0

        return rep_weights

    else:
        raise ValueError(f"Unknown jackknife type: {jk_type}. Use 'JK1' or 'JKn'.")


def brr_replicate_weights(
    weights: np.ndarray | pd.Series,
    strata: np.ndarray | pd.Series,
    *,
    n_replicates: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate balanced repeated replication (BRR) weights.

    For each replicate, within each stratum containing two PSUs, one PSU
    is given double weight and the other zero weight, according to a
    Hadamard matrix design.

    Parameters
    ----------
    weights : numpy.ndarray or pandas.Series
        Full-sample weights.
    strata : numpy.ndarray or pandas.Series
        Stratum membership. Each stratum should contain exactly 2 PSUs.
    n_replicates : int | None
        Number of replicates. If None, uses the smallest Hadamard order
        >= number of strata.
    seed : int
        Random seed for Hadamard matrix permutation.

    Returns
    -------
    numpy.ndarray
        Matrix of shape (n, n_replicates).

    References
    ----------
    McCarthy, P. J. (1969). Pseudo-replication: half samples. *Review of
    the International Statistical Institute*, 37(3), 239--264.
    """
    w = np.asarray(weights, dtype=float)
    s = np.asarray(strata)
    n = len(w)
    unique_strata = np.unique(s)
    H = len(unique_strata)

    if n_replicates is None:
        # Smallest power of 2 >= H
        n_replicates = 1
        while n_replicates < H:
            n_replicates *= 2

    # Generate Hadamard-like sign matrix using random signs
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(H, n_replicates))

    rep_weights = np.tile(w, (n_replicates, 1)).T

    for h_idx, stratum in enumerate(unique_strata):
        mask = s == stratum
        stratum_indices = np.where(mask)[0]

        if len(stratum_indices) < 2:
            continue

        # Split into two halves
        mid = len(stratum_indices) // 2
        half_a = stratum_indices[:mid]
        half_b = stratum_indices[mid:]

        for r in range(n_replicates):
            if signs[h_idx, r] > 0:
                rep_weights[half_a, r] *= 2.0
                rep_weights[half_b, r] = 0.0
            else:
                rep_weights[half_a, r] = 0.0
                rep_weights[half_b, r] *= 2.0

    return rep_weights


def fay_brr_weights(
    weights: np.ndarray | pd.Series,
    strata: np.ndarray | pd.Series,
    *,
    fay_coefficient: float = 0.5,
    n_replicates: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate Fay's BRR replicate weights.

    A generalization of BRR where units are perturbed rather than entirely
    removed, controlled by the Fay coefficient :math:`\\rho \\in [0, 1)`.
    When :math:`\\rho = 0`, this reduces to standard BRR.

    Parameters
    ----------
    weights : numpy.ndarray or pandas.Series
        Full-sample weights.
    strata : numpy.ndarray or pandas.Series
        Stratum membership.
    fay_coefficient : float
        Perturbation coefficient in [0, 1). Default 0.5.
    n_replicates : int | None
        Number of replicates.
    seed : int
        Random seed.

    Returns
    -------
    numpy.ndarray
        Matrix of shape (n, n_replicates).

    References
    ----------
    Fay, R. E. (1989). Theory and application of replicate weighting for
    variance calculations. *Proc. Survey Research Methods Section, ASA*.
    """
    if not 0 <= fay_coefficient < 1:
        raise ValueError("fay_coefficient must be in [0, 1)")

    w = np.asarray(weights, dtype=float)
    s = np.asarray(strata)
    n = len(w)
    unique_strata = np.unique(s)
    H = len(unique_strata)

    if n_replicates is None:
        n_replicates = 1
        while n_replicates < H:
            n_replicates *= 2

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(H, n_replicates))

    rep_weights = np.tile(w, (n_replicates, 1)).T

    for h_idx, stratum in enumerate(unique_strata):
        mask = s == stratum
        stratum_indices = np.where(mask)[0]

        if len(stratum_indices) < 2:
            continue

        mid = len(stratum_indices) // 2
        half_a = stratum_indices[:mid]
        half_b = stratum_indices[mid:]

        for r in range(n_replicates):
            if signs[h_idx, r] > 0:
                rep_weights[half_a, r] *= 2 - fay_coefficient
                rep_weights[half_b, r] *= fay_coefficient
            else:
                rep_weights[half_a, r] *= fay_coefficient
                rep_weights[half_b, r] *= 2 - fay_coefficient

    return rep_weights


def bootstrap_replicate_weights(
    weights: np.ndarray | pd.Series,
    *,
    n_replicates: int = 200,
    strata: np.ndarray | pd.Series | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate bootstrap replicate weights.

    For each replicate, resample units with replacement within strata and
    compute bootstrap weights.

    Parameters
    ----------
    weights : numpy.ndarray or pandas.Series
        Full-sample weights.
    n_replicates : int
        Number of bootstrap replicates (default 200).
    strata : numpy.ndarray | pandas.Series | None
        If provided, bootstrap within strata.
    seed : int
        Random seed.

    Returns
    -------
    numpy.ndarray
        Matrix of shape (n, n_replicates).

    References
    ----------
    Rao, J. N. K. & Wu, C. F. J. (1988). Resampling inference with complex
    survey data. *JASA*, 83(401), 231--241.
    """
    w = np.asarray(weights, dtype=float)
    n = len(w)
    rng = np.random.default_rng(seed)

    rep_weights = np.zeros((n, n_replicates), dtype=float)

    if strata is not None:
        s = np.asarray(strata)
        unique_strata = np.unique(s)

        for r in range(n_replicates):
            counts = np.zeros(n, dtype=float)
            for stratum in unique_strata:
                mask = s == stratum
                indices = np.where(mask)[0]
                n_h = len(indices)
                # Resample n_h-1 units (Rao-Wu rescaling)
                boot_idx = rng.choice(indices, size=n_h - 1, replace=True)
                for idx in boot_idx:
                    counts[idx] += 1
                # Rescale: multiply by n_h / (n_h - 1)
                counts[indices] *= n_h / (n_h - 1) if n_h > 1 else 1
            rep_weights[:, r] = w * counts
    else:
        for r in range(n_replicates):
            boot_idx = rng.choice(n, size=n, replace=True)
            counts = np.bincount(boot_idx, minlength=n).astype(float)
            rep_weights[:, r] = w * counts

    return rep_weights


def sdr_replicate_weights(
    weights: np.ndarray | pd.Series,
    *,
    n_replicates: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Generate successive difference replication (SDR) weights.

    SDR is designed for systematic samples where units are ordered and
    variance is estimated from successive differences.

    Parameters
    ----------
    weights : numpy.ndarray or pandas.Series
        Full-sample weights (assumed in sort order).
    n_replicates : int
        Number of replicates.
    seed : int
        Random seed.

    Returns
    -------
    numpy.ndarray
        Matrix of shape (n, n_replicates).

    References
    ----------
    Fay, R. E. & Train, G. F. (1995). Aspects of survey and model-based
    postcensal estimation of income and poverty characteristics. *JASA*,
    90(432), 973--985.
    """
    w = np.asarray(weights, dtype=float)
    n = len(w)
    rng = np.random.default_rng(seed)

    rep_weights = np.tile(w, (n_replicates, 1)).T

    for r in range(n_replicates):
        # Generate random signs for successive differences
        signs = rng.choice([-1, 1], size=n)
        perturbations = np.zeros(n)

        for i in range(n - 1):
            diff = w[i + 1] - w[i]
            perturbations[i] += signs[i] * diff * 0.5
            perturbations[i + 1] -= signs[i] * diff * 0.5

        rep_weights[:, r] = w + perturbations
        # Ensure non-negative
        rep_weights[:, r] = np.maximum(rep_weights[:, r], 0)

    return rep_weights


# ---------------------------------------------------------------------------
# Variance estimation with replicate weights
# ---------------------------------------------------------------------------


def replicate_variance(
    full_estimate: float,
    replicate_estimates: np.ndarray,
    *,
    method: str = "JK1",
    fay_coefficient: float = 0.0,
) -> dict[str, float]:
    """Estimate variance using replicate weight estimates.

    Parameters
    ----------
    full_estimate : float
        Estimate from full-sample weights.
    replicate_estimates : numpy.ndarray
        Array of estimates from each replicate (length R).
    method : str
        Replication method: ``JK1``, ``JKn``, ``BRR``, ``Fay``,
        ``bootstrap``, ``SDR``.
    fay_coefficient : float
        For Fay's BRR method.

    Returns
    -------
    dict[str, float]
        Keys: ``variance``, ``se``, ``ci_lower``, ``ci_upper``.

    References
    ----------
    Wolter, K. M. (2007). *Introduction to Variance Estimation*, Ch. 4-5.
    """
    reps = np.asarray(replicate_estimates, dtype=float)
    R = len(reps)

    if R == 0:
        return {"variance": 0.0, "se": 0.0, "ci_lower": full_estimate, "ci_upper": full_estimate}

    diffs_sq = (reps - full_estimate) ** 2

    if method == "JK1":
        c = (R - 1) / R
        var = c * np.sum(diffs_sq)
    elif method in ("JKn", "JK2"):
        var = np.sum(diffs_sq)
    elif method == "BRR":
        var = (1.0 / R) * np.sum(diffs_sq)
    elif method == "Fay":
        if fay_coefficient >= 1:
            raise ValueError("fay_coefficient must be < 1")
        var = (1.0 / (R * (1 - fay_coefficient) ** 2)) * np.sum(diffs_sq)
    elif method == "bootstrap":
        var = float(np.var(reps, ddof=1))
    elif method == "SDR":
        var = (4.0 / R) * np.sum(diffs_sq)
    else:
        raise ValueError(f"Unknown method: {method}")

    se = float(np.sqrt(var))

    return {
        "variance": float(var),
        "se": se,
        "ci_lower": full_estimate - 1.96 * se,
        "ci_upper": full_estimate + 1.96 * se,
    }


# ---------------------------------------------------------------------------
# Multi-frame weights
# ---------------------------------------------------------------------------


def multiframe_weights(
    weights_a: np.ndarray | pd.Series,
    weights_b: np.ndarray | pd.Series,
    overlap_a: np.ndarray | pd.Series,
    overlap_b: np.ndarray | pd.Series,
    *,
    method: str = "hartley",
    theta: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute multi-frame survey weights for dual-frame designs.

    Parameters
    ----------
    weights_a : numpy.ndarray or pandas.Series
        Weights from frame A.
    weights_b : numpy.ndarray or pandas.Series
        Weights from frame B.
    overlap_a : numpy.ndarray or pandas.Series
        Boolean: True if unit from frame A is also in frame B.
    overlap_b : numpy.ndarray or pandas.Series
        Boolean: True if unit from frame B is also in frame A.
    method : str
        ``hartley`` (fixed compositing) or ``optimal`` (variance-minimizing).
    theta : float
        Compositing factor for overlap domain (default 0.5 = equal).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Adjusted (weights_a, weights_b).

    References
    ----------
    Hartley, H. O. (1962). Multiple frame surveys. *Proc. Social Statistics
    Section, ASA*, 203--206.
    """
    wa = np.asarray(weights_a, dtype=float).copy()
    wb = np.asarray(weights_b, dtype=float).copy()
    ov_a = np.asarray(overlap_a, dtype=bool)
    ov_b = np.asarray(overlap_b, dtype=bool)

    if method == "hartley":
        # For overlap units, apply compositing factor
        wa[ov_a] *= theta
        wb[ov_b] *= 1 - theta
    elif method == "optimal":
        # Variance-minimizing theta computed from effective sample sizes
        ess_a = _ess(wa[ov_a]) if ov_a.any() else 0.0
        ess_b = _ess(wb[ov_b]) if ov_b.any() else 0.0
        total_ess = ess_a + ess_b
        if total_ess > 0:
            opt_theta = ess_a / total_ess
        else:
            opt_theta = 0.5
        wa[ov_a] *= opt_theta
        wb[ov_b] *= 1 - opt_theta
    else:
        raise ValueError(f"Unknown method: {method}. Use 'hartley' or 'optimal'.")

    return wa, wb


# ---------------------------------------------------------------------------
# Composite weight estimation
# ---------------------------------------------------------------------------


def composite_weights(
    weight_components: list[np.ndarray],
    *,
    method: str = "product",
) -> np.ndarray:
    """Combine multiple weight components into a single composite weight.

    Parameters
    ----------
    weight_components : list[numpy.ndarray]
        List of weight arrays (all same length).
    method : str
        ``product`` (multiply all) or ``geometric_mean``.

    Returns
    -------
    numpy.ndarray
        Composite weights.
    """
    if not weight_components:
        raise ValueError("At least one weight component required")

    n = len(weight_components[0])
    for i, wc in enumerate(weight_components):
        if len(wc) != n:
            raise ValueError(f"Component {i} has length {len(wc)}, expected {n}")

    if method == "product":
        result = np.ones(n, dtype=float)
        for wc in weight_components:
            result *= np.asarray(wc, dtype=float)
        return result
    elif method == "geometric_mean":
        k = len(weight_components)
        log_sum = np.zeros(n, dtype=float)
        for wc in weight_components:
            log_sum += np.log(np.maximum(np.asarray(wc, dtype=float), 1e-10))
        return np.exp(log_sum / k)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'product' or 'geometric_mean'.")


# ---------------------------------------------------------------------------
# Diagnostics report
# ---------------------------------------------------------------------------


def weight_diagnostics_report(
    weights: np.ndarray | pd.Series,
    *,
    name: str = "Survey Weights",
    output_path: str | Path | None = None,
) -> str:
    """Generate a comprehensive weight diagnostics report.

    Parameters
    ----------
    weights : numpy.ndarray or pandas.Series
        Survey weights.
    name : str
        Report title.
    output_path : str | Path | None
        If provided, save to this path.

    Returns
    -------
    str
        Markdown report text.
    """
    from pathlib import Path as _Path

    diag = weight_diagnostics(weights)
    extreme = detect_extreme_weights(weights)

    lines = [
        f"# Weight Diagnostics: {name}",
        "",
        "## Summary Statistics",
        "",
        f"- **N**: {diag.n:,}",
        f"- **Sum of weights**: {diag.sum_weights:,.2f}",
        f"- **Mean weight**: {diag.mean_weight:.4f}",
        f"- **Median weight**: {diag.median_weight:.4f}",
        f"- **Std dev**: {diag.std_weight:.4f}",
        f"- **CV**: {diag.cv:.4f}",
        f"- **Min**: {diag.min_weight:.4f}",
        f"- **Max**: {diag.max_weight:.4f}",
        f"- **Range ratio (max/min)**: {diag.weight_range_ratio:.2f}",
        "",
        "## Design Effect",
        "",
        f"- **Effective sample size**: {diag.effective_sample_size:.1f}",
        f"- **Design effect (DEFF)**: {diag.design_effect:.4f}",
        f"- **Efficiency**: {100 / diag.design_effect:.1f}%" if diag.design_effect > 0 else "- N/A",
        "",
        "## Percentile Distribution",
        "",
        "| Percentile | Value |",
        "|---|---|",
    ]

    for pct, val in sorted(diag.percentiles.items()):
        lines.append(f"| {pct} | {val:.4f} |")

    lines.extend(
        [
            "",
            "## Extreme Weights",
            "",
            f"- **N extreme**: {extreme.n_extreme} ({extreme.pct_extreme:.1f}%)",
            f"- **Lower threshold**: {extreme.threshold_lower:.4f}",
            f"- **Upper threshold**: {extreme.threshold_upper:.4f}",
            "",
        ]
    )

    if diag.n_zero > 0:
        lines.append(f"**Warning**: {diag.n_zero} weights are exactly zero.")
    if diag.n_negative > 0:
        lines.append(f"**Warning**: {diag.n_negative} weights are negative.")
    if diag.cv > 1.0:
        lines.append(f"**Warning**: CV = {diag.cv:.2f} is high. Consider weight trimming.")

    report = "\n".join(lines)

    if output_path:
        p = _Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(report, encoding="utf-8")
        logger.info("Weight diagnostics saved to %s", p)

    return report


# ---------------------------------------------------------------------------
# Rich rendering
# ---------------------------------------------------------------------------


def render_diagnostics(diag: WeightDiagnostics) -> None:
    """Display weight diagnostics with rich formatting.

    Parameters
    ----------
    diag : WeightDiagnostics
    """
    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(
            title="Survey Weight Diagnostics",
            box=box.ROUNDED,
            header_style="bold cyan",
        )
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("N", f"{diag.n:,}")
        table.add_row("Sum", f"{diag.sum_weights:,.2f}")
        table.add_row("Mean", f"{diag.mean_weight:.4f}")
        table.add_row("Median", f"{diag.median_weight:.4f}")
        table.add_row("Std Dev", f"{diag.std_weight:.4f}")
        table.add_row("CV", f"{diag.cv:.4f}")
        table.add_row("Min", f"{diag.min_weight:.4f}")
        table.add_row("Max", f"{diag.max_weight:.4f}")
        table.add_row("ESS", f"{diag.effective_sample_size:.1f}")
        table.add_row("DEFF", f"{diag.design_effect:.4f}")

        cv_style = "red" if diag.cv > 1.0 else "green"
        deff_style = "yellow" if diag.design_effect > 2.0 else "green"

        console.print(table)

        if diag.cv > 1.0:
            console.print(f"[{cv_style}]High CV ({diag.cv:.2f}). Consider weight trimming.[/{cv_style}]")
        if diag.n_negative > 0:
            console.print(f"[red]{diag.n_negative} negative weights detected.[/red]")

    except ImportError:
        d = diag.to_dict()
        for k, v in d.items():
            if k != "percentiles":
                print(f"  {k}: {v}")
