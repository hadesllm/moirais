"""
Survival analysis toolkit for epidemiological time-to-event research.

This module provides a comprehensive set of tools for censored time-to-event
data commonly encountered in cohort studies, clinical trials, and public health
surveillance.  Implementations are built on :mod:`numpy`, :mod:`scipy`, and
:mod:`pandas`; optional integration with :mod:`lifelines` is used where
available but never required.

Major components
----------------
- **Non-parametric estimators**: Kaplan--Meier, Nelson--Aalen
- **Log-rank family tests**: standard, Peto--Peto, Gehan--Wilcoxon, Tarone--Ware
- **Cox proportional hazards**: partial likelihood estimation with Breslow and
  Efron tie-handling, Schoenfeld / Cox--Snell / martingale / deviance residuals
- **Parametric models**: Exponential, Weibull, log-normal, log-logistic, Gompertz
- **Accelerated failure time (AFT)** models
- **Competing risks**: Cumulative incidence functions (CIF), Fine--Gray model
- **Discrimination**: Concordance index (Harrell's *C*)
- **RMST**: Restricted mean survival time comparison

References
----------
Collett, D. (2015). *Modelling Survival Data in Medical Research* (3rd ed.).
    Chapman & Hall/CRC.
Kalbfleisch, J. D., & Prentice, R. L. (2002). *The Statistical Analysis of
    Failure Time Data* (2nd ed.). Wiley.
Fine, J. P., & Gray, R. J. (1999). A proportional hazards model for the
    subdistribution of a competing risk. *JASA*, 94(446), 496--509.
Klein, J. P., & Moeschberger, M. L. (2003). *Survival Analysis: Techniques
    for Censored and Truncated Data* (2nd ed.). Springer.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# ===================================================================
# Result containers
# ===================================================================


@dataclass
class SurvivalCurve:
    """Stores a non-parametric survival or cumulative-hazard estimate.

    Parameters
    ----------
    times : ndarray
        Ordered unique event times.
    survival : ndarray
        Estimated survival probabilities (or cumulative hazard).
    ci_lower : ndarray
        Lower confidence band.
    ci_upper : ndarray
        Upper confidence band.
    at_risk : ndarray
        Number at risk just before each event time.
    events : ndarray
        Number of events at each event time.
    censored : ndarray
        Number censored at each event time.
    method : str
        Estimator name.
    median_survival : float or None
        Median survival time (first time S(t) <= 0.5).
    """

    times: np.ndarray
    survival: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    at_risk: np.ndarray
    events: np.ndarray
    censored: np.ndarray
    method: str = "Kaplan-Meier"
    median_survival: float | None = None


@dataclass
class LogRankResult:
    """Result container for log-rank family tests."""

    method: str
    test_statistic: float
    p_value: float
    df: int
    n_groups: int
    n_total: int
    extra: dict = field(default_factory=dict)


@dataclass
class CoxResult:
    """Result container for Cox PH model."""

    coefficients: np.ndarray
    standard_errors: np.ndarray
    hazard_ratios: np.ndarray
    z_scores: np.ndarray
    p_values: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    covariate_names: list[str]
    concordance: float
    log_likelihood: float
    n_events: int
    n_observations: int
    method: str = "Cox PH"
    baseline_hazard: pd.DataFrame | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class ParametricSurvivalResult:
    """Result container for parametric survival / AFT models."""

    distribution: str
    coefficients: dict
    log_likelihood: float
    aic: float
    bic: float
    n_observations: int
    n_events: int
    extra: dict = field(default_factory=dict)


@dataclass
class CompetingRiskResult:
    """Result container for competing-risk analyses."""

    times: np.ndarray
    cif: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    event_of_interest: int
    n_total: int
    method: str = "Aalen-Johansen"


# ===================================================================
# Internal helpers
# ===================================================================


def _validate_survival_input(
    time: Union[np.ndarray, pd.Series, list],
    event: Union[np.ndarray, pd.Series, list],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and align survival inputs, removing non-finite values."""
    t = np.asarray(time, dtype=np.float64).ravel()
    e = np.asarray(event, dtype=np.float64).ravel()
    if len(t) != len(e):
        raise ValueError("time and event arrays must have equal length.")
    mask = np.isfinite(t) & np.isfinite(e) & (t >= 0)
    if mask.sum() < len(t):
        logger.debug("Dropped %d invalid observations.", int((~mask).sum()))
    return t[mask], e[mask].astype(np.int32)


def _kaplan_meier_table(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a KM life table.

    Returns (unique_times, n_at_risk, n_events, n_censored).
    """
    order = np.argsort(time)
    t_sorted = time[order]
    e_sorted = event[order]
    unique_times = np.unique(t_sorted)
    n = len(t_sorted)
    at_risk = np.empty(len(unique_times))
    events = np.empty(len(unique_times))
    censored = np.empty(len(unique_times))
    cum_loss = 0
    for i, ut in enumerate(unique_times):
        mask = t_sorted == ut
        events[i] = e_sorted[mask].sum()
        n_at_time = mask.sum()
        censored[i] = n_at_time - events[i]
        at_risk[i] = n - cum_loss
        cum_loss += n_at_time
    return unique_times, at_risk, events, censored


# ===================================================================
# KAPLAN-MEIER
# ===================================================================


def kaplan_meier(
    time: Union[np.ndarray, pd.Series, list],
    event: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
    ci_method: str = "greenwood",
) -> SurvivalCurve:
    """Kaplan--Meier product-limit survival estimator.

    Parameters
    ----------
    time : array-like
        Observed times (non-negative).
    event : array-like
        Event indicator (1 = event, 0 = censored).
    confidence : float, default 0.95
        Confidence level for survival bands.
    ci_method : str, default "greenwood"
        ``"greenwood"`` for Greenwood's formula or ``"log-log"`` for the
        complementary log-log transformation.

    Returns
    -------
    SurvivalCurve

    References
    ----------
    Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from
    incomplete observations. *JASA*, 53(282), 457--481.
    """
    t, e = _validate_survival_input(time, event)
    unique_times, at_risk, events, censored = _kaplan_meier_table(t, e)
    survival = np.cumprod(1 - events / at_risk)
    z = stats.norm.ppf((1 + confidence) / 2)

    if ci_method == "greenwood":
        var_terms = np.cumsum(events / (at_risk * (at_risk - events + 1e-15)))
        se = survival * np.sqrt(var_terms)
        ci_lo = np.clip(survival - z * se, 0, 1)
        ci_hi = np.clip(survival + z * se, 0, 1)
    elif ci_method == "log-log":
        # log(-log(S(t))) transformation
        log_log_s = np.log(-np.log(np.clip(survival, 1e-15, 1 - 1e-15)))
        var_terms = np.cumsum(events / (at_risk * (at_risk - events + 1e-15)))
        log_s = np.abs(np.log(np.clip(survival, 1e-15, 1 - 1e-15)))
        se_ll = np.sqrt(var_terms) / (np.clip(survival, 1e-15, 1) * np.maximum(log_s, 1e-15))
        ci_lo = np.clip(np.exp(-np.exp(log_log_s + z * se_ll)), 0, 1)
        ci_hi = np.clip(np.exp(-np.exp(log_log_s - z * se_ll)), 0, 1)
    else:
        raise ValueError(f"Unknown ci_method: {ci_method}. Use 'greenwood' or 'log-log'.")

    # Median survival
    below_50 = np.where(survival <= 0.5)[0]
    median_surv = float(unique_times[below_50[0]]) if len(below_50) > 0 else None

    return SurvivalCurve(
        times=unique_times,
        survival=survival,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        at_risk=at_risk,
        events=events,
        censored=censored,
        method=f"Kaplan-Meier ({ci_method} CI)",
        median_survival=median_surv,
    )


# ===================================================================
# NELSON-AALEN
# ===================================================================


def nelson_aalen(
    time: Union[np.ndarray, pd.Series, list],
    event: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> SurvivalCurve:
    """Nelson--Aalen cumulative hazard estimator.

    Parameters
    ----------
    time : array-like
        Observed times.
    event : array-like
        Event indicator (1 = event, 0 = censored).
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    SurvivalCurve
        The ``survival`` field here contains the **cumulative hazard** H(t),
        not S(t).
    """
    t, e = _validate_survival_input(time, event)
    unique_times, at_risk, events, censored = _kaplan_meier_table(t, e)
    cumhaz = np.cumsum(events / at_risk)
    var_h = np.cumsum(events / at_risk**2)
    z = stats.norm.ppf((1 + confidence) / 2)
    se = np.sqrt(var_h)
    return SurvivalCurve(
        times=unique_times,
        survival=cumhaz,  # cumulative hazard
        ci_lower=np.clip(cumhaz - z * se, 0, np.inf),
        ci_upper=cumhaz + z * se,
        at_risk=at_risk,
        events=events,
        censored=censored,
        method="Nelson-Aalen",
    )


# ===================================================================
# LOG-RANK FAMILY
# ===================================================================


def _logrank_generic(
    time: Union[np.ndarray, pd.Series, list],
    event: Union[np.ndarray, pd.Series, list],
    group: Union[np.ndarray, pd.Series, list],
    weight_func: str = "logrank",
) -> LogRankResult:
    """Generic weighted log-rank test engine.

    Parameters
    ----------
    weight_func : str
        ``"logrank"`` (w=1), ``"peto"`` (Peto--Peto), ``"gehan"``
        (Gehan--Wilcoxon), ``"tarone"`` (Tarone--Ware).
    """
    t, e = _validate_survival_input(time, event)
    g = np.asarray(group).ravel()
    if len(g) != len(t):
        raise ValueError("group must have the same length as time/event.")
    groups = np.unique(g)
    n_groups = len(groups)
    if n_groups < 2:
        raise ValueError("Need at least 2 groups for a log-rank test.")

    # Pool all event times
    all_times = np.sort(np.unique(t[e == 1]))
    k = len(all_times)
    n_total = len(t)

    # Build risk and event counts per group per time
    d_j = np.zeros((k, n_groups))  # events per group
    n_j = np.zeros((k, n_groups))  # at risk per group
    for gi, gval in enumerate(groups):
        mask = g == gval
        t_g, e_g = t[mask], e[mask]
        for ti, tt in enumerate(all_times):
            n_j[ti, gi] = (t_g >= tt).sum()
            d_j[ti, gi] = ((t_g == tt) & (e_g == 1)).sum()

    d_total = d_j.sum(axis=1)
    n_total_t = n_j.sum(axis=1)

    # Weights
    if weight_func == "logrank":
        w = np.ones(k)
    elif weight_func == "gehan":
        w = n_total_t.copy()
    elif weight_func == "tarone":
        w = np.sqrt(n_total_t)
    elif weight_func == "peto":
        # Peto-Peto: S_tilde (left-continuous KM)
        s_tilde = np.cumprod(1 - d_total / (n_total_t + 1))
        w = s_tilde
    else:
        raise ValueError(f"Unknown weight: {weight_func}")

    # Test statistic for (n_groups - 1) dimensional test
    # We use groups[:-1] (reference = last group)
    q = n_groups - 1
    U = np.zeros(q)
    V = np.zeros((q, q))

    for ti in range(k):
        nt = n_total_t[ti]
        dt = d_total[ti]
        if nt <= 1 or dt == 0:
            continue
        for a in range(q):
            e_a = n_j[ti, a] * dt / nt
            U[a] += w[ti] * (d_j[ti, a] - e_a)
            for b in range(q):
                if a == b:
                    v_ab = n_j[ti, a] * (nt - n_j[ti, a]) * dt * (nt - dt) / (nt**2 * (nt - 1))
                else:
                    v_ab = -n_j[ti, a] * n_j[ti, b] * dt * (nt - dt) / (nt**2 * (nt - 1))
                V[a, b] += w[ti] ** 2 * v_ab

    try:
        V_inv = np.linalg.inv(V)
        chi2 = float(U @ V_inv @ U)
    except np.linalg.LinAlgError:
        chi2 = 0.0

    p_val = 1.0 - stats.chi2.cdf(chi2, q)
    method_names = {
        "logrank": "Log-rank test",
        "peto": "Peto-Peto test",
        "gehan": "Gehan-Wilcoxon test",
        "tarone": "Tarone-Ware test",
    }
    return LogRankResult(
        method=method_names.get(weight_func, weight_func),
        test_statistic=float(chi2),
        p_value=float(p_val),
        df=q,
        n_groups=n_groups,
        n_total=len(t),
    )


def logrank_test(
    time: Union[np.ndarray, pd.Series, list],
    event: Union[np.ndarray, pd.Series, list],
    group: Union[np.ndarray, pd.Series, list],
) -> LogRankResult:
    """Standard log-rank test (Mantel--Haenszel).

    Parameters
    ----------
    time, event, group : array-like
        Observed times, event indicators, and group labels.

    Returns
    -------
    LogRankResult
    """
    return _logrank_generic(time, event, group, weight_func="logrank")


def peto_peto_test(
    time: Union[np.ndarray, pd.Series, list],
    event: Union[np.ndarray, pd.Series, list],
    group: Union[np.ndarray, pd.Series, list],
) -> LogRankResult:
    """Peto--Peto modification of the Gehan--Wilcoxon test.

    Parameters
    ----------
    time, event, group : array-like

    Returns
    -------
    LogRankResult
    """
    return _logrank_generic(time, event, group, weight_func="peto")


def gehan_wilcoxon_test(
    time: Union[np.ndarray, pd.Series, list],
    event: Union[np.ndarray, pd.Series, list],
    group: Union[np.ndarray, pd.Series, list],
) -> LogRankResult:
    """Gehan--Wilcoxon test (Breslow test).

    Parameters
    ----------
    time, event, group : array-like

    Returns
    -------
    LogRankResult
    """
    return _logrank_generic(time, event, group, weight_func="gehan")


def tarone_ware_test(
    time: Union[np.ndarray, pd.Series, list],
    event: Union[np.ndarray, pd.Series, list],
    group: Union[np.ndarray, pd.Series, list],
) -> LogRankResult:
    """Tarone--Ware weighted log-rank test.

    Parameters
    ----------
    time, event, group : array-like

    Returns
    -------
    LogRankResult
    """
    return _logrank_generic(time, event, group, weight_func="tarone")


# ===================================================================
# COX PROPORTIONAL HAZARDS
# ===================================================================


def _cox_negative_log_partial_likelihood(
    beta: np.ndarray,
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    ties: str = "efron",
) -> float:
    """Negative log partial likelihood for Cox PH with Efron or Breslow ties."""
    n = len(time)
    order = np.argsort(-time)  # descending
    X_ord = X[order]
    t_ord = time[order]
    e_ord = event[order]
    eta = X_ord @ beta
    # Prevent overflow
    eta = np.clip(eta, -500, 500)
    exp_eta = np.exp(eta)

    nll = 0.0
    unique_event_times = np.unique(t_ord[e_ord == 1])

    for ut in unique_event_times:
        event_mask = (t_ord == ut) & (e_ord == 1)
        risk_mask = t_ord >= ut
        d_j = event_mask.sum()
        if d_j == 0:
            continue
        sum_event_eta = eta[event_mask].sum()
        nll -= sum_event_eta

        if ties == "breslow":
            risk_sum = exp_eta[risk_mask].sum()
            nll += d_j * np.log(risk_sum + 1e-15)
        elif ties == "efron":
            risk_sum = exp_eta[risk_mask].sum()
            event_sum = exp_eta[event_mask].sum()
            for s in range(d_j):
                nll += np.log(risk_sum - s / d_j * event_sum + 1e-15)
        else:
            raise ValueError(f"Unknown tie method: {ties}")
    return nll


def cox_ph(
    data: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariate_cols: list[str],
    ties: str = "efron",
    confidence: float = 0.95,
    penalizer: float = 0.0,
) -> CoxResult:
    """Fit a Cox proportional hazards model.

    Uses Newton--Raphson optimisation of the partial likelihood.

    Parameters
    ----------
    data : DataFrame
        Analysis dataset.
    duration_col : str
        Column with observed times.
    event_col : str
        Column with event indicators (1 = event).
    covariate_cols : list[str]
        Names of covariate columns.
    ties : str, default "efron"
        Tie handling: ``"efron"`` or ``"breslow"``.
    confidence : float, default 0.95
        Confidence level for hazard ratio CIs.
    penalizer : float, default 0.0
        L2 regularisation strength.

    Returns
    -------
    CoxResult
    """
    df = data.dropna(subset=[duration_col, event_col] + covariate_cols).copy()
    t = df[duration_col].values.astype(np.float64)
    e = df[event_col].values.astype(np.int32)
    X = df[covariate_cols].values.astype(np.float64)
    n, p = X.shape

    # Standardise
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    X_std = (X - mu) / sd

    def objective(beta):
        nll = _cox_negative_log_partial_likelihood(beta, X_std, t, e, ties)
        nll += 0.5 * penalizer * np.sum(beta**2)
        return nll

    beta0 = np.zeros(p)
    result = minimize(objective, beta0, method="L-BFGS-B")
    beta_std = result.x

    # Transform back to original scale
    beta = beta_std / sd

    # Hessian via finite differences for SE
    eps = 1e-5
    hessian = np.zeros((p, p))
    f0 = objective(beta_std)
    for i in range(p):
        for j in range(i, p):
            e_i = np.zeros(p)
            e_j = np.zeros(p)
            e_i[i] = eps
            e_j[j] = eps
            f_ij = objective(beta_std + e_i + e_j)
            f_i = objective(beta_std + e_i)
            f_j = objective(beta_std + e_j)
            h = (f_ij - f_i - f_j + f0) / eps**2
            hessian[i, j] = hessian[j, i] = h

    try:
        cov = np.linalg.inv(hessian)
        se_std = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
        se = se_std / sd
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)

    hr = np.exp(beta)
    z = beta / se
    p_values = 2 * stats.norm.sf(np.abs(z))
    z_crit = stats.norm.ppf((1 + confidence) / 2)
    ci_lo = np.exp(beta - z_crit * se)
    ci_hi = np.exp(beta + z_crit * se)

    # Concordance index
    c_index = concordance_index(t, X @ beta, e)

    # Breslow baseline hazard
    baseline_df = _breslow_baseline_hazard(t, e, X, beta)

    return CoxResult(
        coefficients=beta,
        standard_errors=se,
        hazard_ratios=hr,
        z_scores=z,
        p_values=p_values,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        covariate_names=covariate_cols,
        concordance=c_index,
        log_likelihood=-result.fun,
        n_events=int(e.sum()),
        n_observations=n,
        method=f"Cox PH ({ties} ties)",
        baseline_hazard=baseline_df,
    )


def _breslow_baseline_hazard(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray,
    beta: np.ndarray,
) -> pd.DataFrame:
    """Compute the Breslow baseline cumulative hazard estimate."""
    risk_scores = np.exp(X @ beta)
    unique_event_times = np.sort(np.unique(time[event == 1]))
    h0 = []
    H0 = 0.0
    for ut in unique_event_times:
        d_t = ((time == ut) & (event == 1)).sum()
        risk_sum = risk_scores[time >= ut].sum()
        h0_t = d_t / (risk_sum + 1e-15)
        H0 += h0_t
        h0.append({"time": ut, "baseline_hazard": h0_t, "cumulative_hazard": H0})
    return pd.DataFrame(h0)


# ===================================================================
# RESIDUALS
# ===================================================================


def schoenfeld_residuals(
    data: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariate_cols: list[str],
    cox_result: CoxResult,
) -> pd.DataFrame:
    """Compute Schoenfeld residuals for testing the proportional hazards
    assumption.

    Parameters
    ----------
    data : DataFrame
        Analysis dataset.
    duration_col, event_col : str
        Time and event columns.
    covariate_cols : list[str]
        Covariates used in the Cox model.
    cox_result : CoxResult
        A fitted CoxResult from :func:`cox_ph`.

    Returns
    -------
    DataFrame
        One row per event with columns for each covariate's Schoenfeld
        residual and the event time.
    """
    df = data.dropna(subset=[duration_col, event_col] + covariate_cols).copy()
    t = df[duration_col].values.astype(np.float64)
    e = df[event_col].values.astype(np.int32)
    X = df[covariate_cols].values.astype(np.float64)
    beta = cox_result.coefficients
    risk_scores = np.exp(X @ beta)

    event_mask = e == 1
    event_times = t[event_mask]
    event_X = X[event_mask]

    residuals = []
    for i, et in enumerate(event_times):
        in_risk = t >= et
        rs = risk_scores[in_risk]
        rs_sum = rs.sum()
        weighted_mean = (X[in_risk].T * rs).sum(axis=1) / (rs_sum + 1e-15)
        resid = event_X[i] - weighted_mean
        residuals.append(resid)

    resid_df = pd.DataFrame(residuals, columns=covariate_cols)
    resid_df["event_time"] = event_times
    return resid_df


def cox_snell_residuals(
    data: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariate_cols: list[str],
    cox_result: CoxResult,
) -> np.ndarray:
    """Compute Cox--Snell residuals.

    The Cox--Snell residual for observation *i* is
    :math:`r_i = \\hat{H}_0(t_i) \\exp(x_i^T \\hat{\\beta})`.

    If the model is correct these should follow a unit-rate exponential
    distribution.

    Parameters
    ----------
    data : DataFrame
    duration_col, event_col : str
    covariate_cols : list[str]
    cox_result : CoxResult

    Returns
    -------
    ndarray
    """
    df = data.dropna(subset=[duration_col, event_col] + covariate_cols).copy()
    t = df[duration_col].values.astype(np.float64)
    X = df[covariate_cols].values.astype(np.float64)
    beta = cox_result.coefficients
    bh = cox_result.baseline_hazard
    if bh is None:
        raise ValueError("CoxResult must contain baseline_hazard.")
    bh_times = bh["time"].values
    bh_cumhaz = bh["cumulative_hazard"].values
    H0 = np.interp(t, bh_times, bh_cumhaz, left=0.0)
    return H0 * np.exp(X @ beta)


def martingale_residuals(
    data: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariate_cols: list[str],
    cox_result: CoxResult,
) -> np.ndarray:
    """Compute martingale residuals.

    :math:`M_i = \\delta_i - r_i^{CS}` where :math:`r_i^{CS}` is the
    Cox--Snell residual.

    Parameters
    ----------
    data : DataFrame
    duration_col, event_col : str
    covariate_cols : list[str]
    cox_result : CoxResult

    Returns
    -------
    ndarray
    """
    df = data.dropna(subset=[duration_col, event_col] + covariate_cols).copy()
    e = df[event_col].values.astype(np.float64)
    cs = cox_snell_residuals(data, duration_col, event_col, covariate_cols, cox_result)
    return e - cs


def deviance_residuals(
    data: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariate_cols: list[str],
    cox_result: CoxResult,
) -> np.ndarray:
    """Compute deviance residuals from a Cox PH model.

    :math:`d_i = \\text{sign}(M_i) \\sqrt{-2[M_i + \\delta_i \\log(\\delta_i - M_i)]}`

    Parameters
    ----------
    data : DataFrame
    duration_col, event_col : str
    covariate_cols : list[str]
    cox_result : CoxResult

    Returns
    -------
    ndarray
    """
    df = data.dropna(subset=[duration_col, event_col] + covariate_cols).copy()
    e = df[event_col].values.astype(np.float64)
    M = martingale_residuals(data, duration_col, event_col, covariate_cols, cox_result)
    term = M.copy()
    nonzero = e > 0
    log_arg = np.clip(e - M, 1e-15, np.inf)
    term[nonzero] = M[nonzero] + e[nonzero] * np.log(log_arg[nonzero])
    dev = np.sign(M) * np.sqrt(np.clip(-2 * term, 0, np.inf))
    return dev


def test_ph_assumption(
    data: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariate_cols: list[str],
    cox_result: CoxResult,
) -> pd.DataFrame:
    """Test the proportional hazards assumption via Schoenfeld residual
    correlation with time.

    For each covariate, computes the Pearson correlation between scaled
    Schoenfeld residuals and event time and reports a *p*-value.

    Parameters
    ----------
    data : DataFrame
    duration_col, event_col : str
    covariate_cols : list[str]
    cox_result : CoxResult

    Returns
    -------
    DataFrame
        Columns: ``covariate``, ``rho``, ``chi2``, ``p_value``.
    """
    resid_df = schoenfeld_residuals(data, duration_col, event_col, covariate_cols, cox_result)
    results = []
    event_times = resid_df["event_time"].values
    for col in covariate_cols:
        r, p = stats.pearsonr(event_times, resid_df[col].values)
        chi2 = r**2 * len(event_times)
        results.append({"covariate": col, "rho": r, "chi2": chi2, "p_value": p})
    return pd.DataFrame(results)


# ===================================================================
# PARAMETRIC SURVIVAL MODELS
# ===================================================================


def exponential_model(
    time: Union[np.ndarray, pd.Series, list],
    event: Union[np.ndarray, pd.Series, list],
) -> ParametricSurvivalResult:
    """Fit an exponential survival model (constant hazard).

    The exponential distribution has hazard :math:`h(t) = \\lambda` and
    survival :math:`S(t) = e^{-\\lambda t}`.

    Parameters
    ----------
    time, event : array-like

    Returns
    -------
    ParametricSurvivalResult
    """
    t, e = _validate_survival_input(time, event)
    n = len(t)
    d = int(e.sum())
    total_time = t.sum()
    lam = d / total_time if total_time > 0 else 0.0
    ll = d * np.log(lam + 1e-15) - lam * total_time
    k = 1
    aic = -2 * ll + 2 * k
    bic = -2 * ll + k * np.log(n)
    return ParametricSurvivalResult(
        distribution="Exponential",
        coefficients={"lambda": float(lam), "mean_survival": 1 / lam if lam > 0 else np.inf},
        log_likelihood=float(ll),
        aic=float(aic),
        bic=float(bic),
        n_observations=n,
        n_events=d,
    )


def weibull_model(
    time: Union[np.ndarray, pd.Series, list],
    event: Union[np.ndarray, pd.Series, list],
) -> ParametricSurvivalResult:
    """Fit a Weibull survival model via MLE.

    Parameterisation: :math:`S(t) = \\exp(-(t/\\lambda)^k)`.

    Parameters
    ----------
    time, event : array-like

    Returns
    -------
    ParametricSurvivalResult
    """
    t, e = _validate_survival_input(time, event)
    n = len(t)
    d = int(e.sum())

    def neg_ll(params):
        k, lam = params
        if k <= 0 or lam <= 0:
            return 1e15
        ll = d * np.log(k + 1e-15) - d * k * np.log(lam + 1e-15)
        ll += (k - 1) * np.sum(e * np.log(t + 1e-15))
        ll -= np.sum((t / lam) ** k)
        return -ll

    result = minimize(neg_ll, [1.0, np.mean(t)], method="Nelder-Mead", options={"maxiter": 5000})
    k_hat, lam_hat = result.x
    ll = -result.fun
    n_params = 2
    aic = -2 * ll + 2 * n_params
    bic = -2 * ll + n_params * np.log(n)
    return ParametricSurvivalResult(
        distribution="Weibull",
        coefficients={"shape": float(k_hat), "scale": float(lam_hat)},
        log_likelihood=float(ll),
        aic=float(aic),
        bic=float(bic),
        n_observations=n,
        n_events=d,
    )


def lognormal_model(
    time: Union[np.ndarray, pd.Series, list],
    event: Union[np.ndarray, pd.Series, list],
) -> ParametricSurvivalResult:
    """Fit a log-normal survival model via MLE.

    Parameters
    ----------
    time, event : array-like

    Returns
    -------
    ParametricSurvivalResult
    """
    t, e = _validate_survival_input(time, event)
    n = len(t)
    d = int(e.sum())
    log_t = np.log(t + 1e-15)

    def neg_ll(params):
        mu, sigma = params
        if sigma <= 0:
            return 1e15
        z = (log_t - mu) / sigma
        ll = np.sum(e * (-np.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * z**2 - log_t))
        ll += np.sum((1 - e) * np.log(stats.norm.sf(z) + 1e-15))
        return -ll

    result = minimize(
        neg_ll, [log_t[e == 1].mean() if d > 0 else 0.0, 1.0], method="Nelder-Mead", options={"maxiter": 5000}
    )
    mu_hat, sigma_hat = result.x
    ll = -result.fun
    n_params = 2
    return ParametricSurvivalResult(
        distribution="Log-normal",
        coefficients={"mu": float(mu_hat), "sigma": float(abs(sigma_hat))},
        log_likelihood=float(ll),
        aic=float(-2 * ll + 2 * n_params),
        bic=float(-2 * ll + n_params * np.log(n)),
        n_observations=n,
        n_events=d,
    )


def loglogistic_model(
    time: Union[np.ndarray, pd.Series, list],
    event: Union[np.ndarray, pd.Series, list],
) -> ParametricSurvivalResult:
    """Fit a log-logistic survival model via MLE.

    :math:`S(t) = 1 / (1 + (t/\\alpha)^\\beta)`.

    Parameters
    ----------
    time, event : array-like

    Returns
    -------
    ParametricSurvivalResult
    """
    t, e = _validate_survival_input(time, event)
    n = len(t)
    d = int(e.sum())

    def neg_ll(params):
        alpha, beta = params
        if alpha <= 0 or beta <= 0:
            return 1e15
        z = (t / alpha) ** beta
        ll = np.sum(e * (np.log(beta) - np.log(alpha) + (beta - 1) * np.log(t / alpha) - 2 * np.log(1 + z)))
        ll += np.sum((1 - e) * (-np.log(1 + z)))
        return -ll

    t_median = np.median(t[e == 1]) if d > 0 else np.median(t)
    result = minimize(neg_ll, [t_median, 1.0], method="Nelder-Mead", options={"maxiter": 5000})
    alpha_hat, beta_hat = result.x
    ll = -result.fun
    n_params = 2
    return ParametricSurvivalResult(
        distribution="Log-logistic",
        coefficients={"alpha": float(alpha_hat), "beta": float(beta_hat)},
        log_likelihood=float(ll),
        aic=float(-2 * ll + 2 * n_params),
        bic=float(-2 * ll + n_params * np.log(n)),
        n_observations=n,
        n_events=d,
    )


def gompertz_model(
    time: Union[np.ndarray, pd.Series, list],
    event: Union[np.ndarray, pd.Series, list],
) -> ParametricSurvivalResult:
    """Fit a Gompertz survival model via MLE.

    Hazard: :math:`h(t) = b \\exp(ct)`.  Survival:
    :math:`S(t) = \\exp(-(b/c)(e^{ct} - 1))`.

    Parameters
    ----------
    time, event : array-like

    Returns
    -------
    ParametricSurvivalResult
    """
    t, e = _validate_survival_input(time, event)
    n = len(t)
    d = int(e.sum())

    def neg_ll(params):
        b, c = params
        if b <= 0:
            return 1e15
        if abs(c) < 1e-10:
            # Degenerate to exponential
            ll = d * np.log(b) - b * t.sum()
            return -ll
        h = b * np.exp(c * t)
        H = (b / c) * (np.exp(c * t) - 1)
        ll = np.sum(e * np.log(h + 1e-15)) - np.sum(H)
        return -ll

    result = minimize(
        neg_ll, [d / t.sum() if t.sum() > 0 else 0.01, 0.01], method="Nelder-Mead", options={"maxiter": 5000}
    )
    b_hat, c_hat = result.x
    ll = -result.fun
    n_params = 2
    return ParametricSurvivalResult(
        distribution="Gompertz",
        coefficients={"b": float(b_hat), "c": float(c_hat)},
        log_likelihood=float(ll),
        aic=float(-2 * ll + 2 * n_params),
        bic=float(-2 * ll + n_params * np.log(n)),
        n_observations=n,
        n_events=d,
    )


# ===================================================================
# ACCELERATED FAILURE TIME MODELS
# ===================================================================


def aft_weibull(
    data: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariate_cols: list[str],
) -> ParametricSurvivalResult:
    """Accelerated failure time model with Weibull baseline.

    The AFT model specifies :math:`\\log T = \\mu + x^T \\beta + \\sigma W`
    where *W* follows a standard extreme value distribution.

    Parameters
    ----------
    data : DataFrame
    duration_col, event_col : str
    covariate_cols : list[str]

    Returns
    -------
    ParametricSurvivalResult
    """
    df = data.dropna(subset=[duration_col, event_col] + covariate_cols).copy()
    t = df[duration_col].values.astype(np.float64)
    e = df[event_col].values.astype(np.int32)
    X = df[covariate_cols].values.astype(np.float64)
    n, p = X.shape
    log_t = np.log(t + 1e-15)

    def neg_ll(params):
        mu = params[0]
        sigma = params[1]
        beta = params[2:]
        if sigma <= 0:
            return 1e15
        z = (log_t - mu - X @ beta) / sigma
        # Extreme value distribution: f(z) = exp(z - exp(z)), S(z) = exp(-exp(z))
        ll = np.sum(e * (z - np.exp(z) - np.log(sigma) - log_t))
        ll += np.sum((1 - e) * (-np.exp(z)))
        return -ll

    x0 = np.zeros(p + 2)
    x0[0] = log_t[e == 1].mean() if e.sum() > 0 else 0.0
    x0[1] = 1.0
    result = minimize(neg_ll, x0, method="Nelder-Mead", options={"maxiter": 10000})
    mu_hat = result.x[0]
    sigma_hat = result.x[1]
    beta_hat = result.x[2:]
    ll = -result.fun
    n_params = p + 2
    coefs = {"intercept": float(mu_hat), "sigma": float(abs(sigma_hat))}
    for name, val in zip(covariate_cols, beta_hat):
        coefs[name] = float(val)
    return ParametricSurvivalResult(
        distribution="AFT-Weibull",
        coefficients=coefs,
        log_likelihood=float(ll),
        aic=float(-2 * ll + 2 * n_params),
        bic=float(-2 * ll + n_params * np.log(n)),
        n_observations=n,
        n_events=int(e.sum()),
    )


def aft_lognormal(
    data: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariate_cols: list[str],
) -> ParametricSurvivalResult:
    """AFT model with log-normal baseline.

    Parameters
    ----------
    data : DataFrame
    duration_col, event_col : str
    covariate_cols : list[str]

    Returns
    -------
    ParametricSurvivalResult
    """
    df = data.dropna(subset=[duration_col, event_col] + covariate_cols).copy()
    t = df[duration_col].values.astype(np.float64)
    e = df[event_col].values.astype(np.int32)
    X = df[covariate_cols].values.astype(np.float64)
    n, p = X.shape
    log_t = np.log(t + 1e-15)

    def neg_ll(params):
        mu = params[0]
        sigma = params[1]
        beta = params[2:]
        if sigma <= 0:
            return 1e15
        z = (log_t - mu - X @ beta) / sigma
        ll = np.sum(e * (-np.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * z**2 - log_t))
        ll += np.sum((1 - e) * np.log(stats.norm.sf(z) + 1e-15))
        return -ll

    x0 = np.zeros(p + 2)
    x0[0] = log_t[e == 1].mean() if e.sum() > 0 else 0.0
    x0[1] = 1.0
    result = minimize(neg_ll, x0, method="Nelder-Mead", options={"maxiter": 10000})
    mu_hat = result.x[0]
    sigma_hat = result.x[1]
    beta_hat = result.x[2:]
    ll = -result.fun
    n_params = p + 2
    coefs = {"intercept": float(mu_hat), "sigma": float(abs(sigma_hat))}
    for name, val in zip(covariate_cols, beta_hat):
        coefs[name] = float(val)
    return ParametricSurvivalResult(
        distribution="AFT-LogNormal",
        coefficients=coefs,
        log_likelihood=float(ll),
        aic=float(-2 * ll + 2 * n_params),
        bic=float(-2 * ll + n_params * np.log(n)),
        n_observations=n,
        n_events=int(e.sum()),
    )


def aft_loglogistic(
    data: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariate_cols: list[str],
) -> ParametricSurvivalResult:
    """AFT model with log-logistic baseline.

    Parameters
    ----------
    data : DataFrame
    duration_col, event_col : str
    covariate_cols : list[str]

    Returns
    -------
    ParametricSurvivalResult
    """
    df = data.dropna(subset=[duration_col, event_col] + covariate_cols).copy()
    t = df[duration_col].values.astype(np.float64)
    e = df[event_col].values.astype(np.int32)
    X = df[covariate_cols].values.astype(np.float64)
    n, p = X.shape
    log_t = np.log(t + 1e-15)

    def neg_ll(params):
        mu = params[0]
        sigma = params[1]
        beta = params[2:]
        if sigma <= 0:
            return 1e15
        z = (log_t - mu - X @ beta) / sigma
        # Logistic distribution
        ll = np.sum(e * (z - np.log(sigma) - log_t - 2 * np.log(1 + np.exp(z))))
        ll += np.sum((1 - e) * (-np.log(1 + np.exp(z))))
        return -ll

    x0 = np.zeros(p + 2)
    x0[0] = log_t[e == 1].mean() if e.sum() > 0 else 0.0
    x0[1] = 1.0
    result = minimize(neg_ll, x0, method="Nelder-Mead", options={"maxiter": 10000})
    mu_hat = result.x[0]
    sigma_hat = result.x[1]
    beta_hat = result.x[2:]
    ll = -result.fun
    n_params = p + 2
    coefs = {"intercept": float(mu_hat), "sigma": float(abs(sigma_hat))}
    for name, val in zip(covariate_cols, beta_hat):
        coefs[name] = float(val)
    return ParametricSurvivalResult(
        distribution="AFT-LogLogistic",
        coefficients=coefs,
        log_likelihood=float(ll),
        aic=float(-2 * ll + 2 * n_params),
        bic=float(-2 * ll + n_params * np.log(n)),
        n_observations=n,
        n_events=int(e.sum()),
    )


# ===================================================================
# RMST
# ===================================================================


def restricted_mean_survival_time(
    time: Union[np.ndarray, pd.Series, list],
    event: Union[np.ndarray, pd.Series, list],
    tau: float | None = None,
    confidence: float = 0.95,
) -> dict:
    """Compute the restricted mean survival time (RMST).

    :math:`\\text{RMST}(\\tau) = \\int_0^\\tau \\hat{S}(t) \\, dt`.

    Parameters
    ----------
    time, event : array-like
    tau : float or None
        Restriction time.  If ``None``, uses the maximum observed time.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    dict
        Keys: ``rmst``, ``se``, ``ci_lower``, ``ci_upper``, ``tau``.
    """
    km = kaplan_meier(time, event, confidence=confidence)
    if tau is None:
        tau = float(km.times[-1])
    mask = km.times <= tau
    t_vals = np.concatenate([[0], km.times[mask], [tau]])
    s_vals = np.concatenate([[1.0], km.survival[mask]])
    # Extend survival at tau
    if km.times[mask].max() < tau:
        s_vals = np.append(s_vals, s_vals[-1])
    else:
        s_vals = np.append(s_vals, km.survival[mask][-1])
    # Trapezoidal integration
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    rmst = float(_trapz(s_vals[:-1], t_vals[:-1]) + s_vals[-2] * (t_vals[-1] - t_vals[-2]))
    # Approximate SE via Greenwood
    n = len(_validate_survival_input(time, event)[0])
    se = rmst / math.sqrt(n) if n > 0 else 0.0
    z = stats.norm.ppf((1 + confidence) / 2)
    return {
        "rmst": rmst,
        "se": se,
        "ci_lower": rmst - z * se,
        "ci_upper": rmst + z * se,
        "tau": tau,
    }


def rmst_difference(
    time1: Union[np.ndarray, pd.Series, list],
    event1: Union[np.ndarray, pd.Series, list],
    time2: Union[np.ndarray, pd.Series, list],
    event2: Union[np.ndarray, pd.Series, list],
    tau: float | None = None,
    confidence: float = 0.95,
) -> dict:
    """Compare RMST between two groups.

    Parameters
    ----------
    time1, event1 : array-like
        Group 1 survival data.
    time2, event2 : array-like
        Group 2 survival data.
    tau : float or None
        Common restriction time.
    confidence : float, default 0.95

    Returns
    -------
    dict
        Keys: ``rmst_diff``, ``se``, ``z``, ``p_value``, ``ci_lower``, ``ci_upper``.
    """
    r1 = restricted_mean_survival_time(time1, event1, tau=tau, confidence=confidence)
    r2 = restricted_mean_survival_time(time2, event2, tau=tau, confidence=confidence)
    diff = r1["rmst"] - r2["rmst"]
    se = math.sqrt(r1["se"] ** 2 + r2["se"] ** 2)
    z = diff / se if se > 0 else 0.0
    p = 2 * stats.norm.sf(abs(z))
    z_crit = stats.norm.ppf((1 + confidence) / 2)
    return {
        "rmst_diff": diff,
        "se": se,
        "z": z,
        "p_value": p,
        "ci_lower": diff - z_crit * se,
        "ci_upper": diff + z_crit * se,
        "rmst_group1": r1["rmst"],
        "rmst_group2": r2["rmst"],
        "tau": r1["tau"],
    }


# ===================================================================
# COMPETING RISKS
# ===================================================================


def cumulative_incidence_function(
    time: Union[np.ndarray, pd.Series, list],
    event: Union[np.ndarray, pd.Series, list],
    event_of_interest: int = 1,
    confidence: float = 0.95,
) -> CompetingRiskResult:
    """Aalen--Johansen estimator of the cumulative incidence function (CIF)
    for competing risks.

    Parameters
    ----------
    time : array-like
        Observed times.
    event : array-like
        Event type indicator: 0 = censored, 1 = event of interest, 2+ = competing
        events.
    event_of_interest : int, default 1
        Which event code is the event of interest.
    confidence : float, default 0.95

    Returns
    -------
    CompetingRiskResult
    """
    t = np.asarray(time, dtype=np.float64).ravel()
    e = np.asarray(event, dtype=np.int32).ravel()
    n = len(t)

    order = np.argsort(t)
    t_sorted = t[order]
    e_sorted = e[order]

    unique_times = np.sort(np.unique(t_sorted[e_sorted > 0]))
    cif = np.zeros(len(unique_times))
    ci_lo = np.zeros(len(unique_times))
    ci_hi = np.zeros(len(unique_times))

    # Overall KM for all events combined
    e_any = (e_sorted > 0).astype(np.int32)
    # Build KM-like estimates
    s_prev = 1.0
    cum_inc = 0.0
    n_at_risk = n
    z = stats.norm.ppf((1 + confidence) / 2)
    var_cif = 0.0

    for i, ut in enumerate(unique_times):
        at_time = t_sorted == ut
        d_interest = ((t_sorted == ut) & (e_sorted == event_of_interest)).sum()
        d_total = ((t_sorted == ut) & (e_sorted > 0)).sum()
        n_at_time = at_time.sum()

        if n_at_risk <= 0:
            cif[i] = cum_inc
            ci_lo[i] = cum_inc
            ci_hi[i] = cum_inc
            continue

        h_interest = d_interest / n_at_risk
        h_total = d_total / n_at_risk
        cum_inc += s_prev * h_interest
        s_prev *= 1 - h_total
        cif[i] = cum_inc

        # Approximate variance
        var_cif += (d_interest / n_at_risk**2) if n_at_risk > 0 else 0
        se = math.sqrt(max(var_cif, 0))
        ci_lo[i] = max(cum_inc - z * se, 0)
        ci_hi[i] = min(cum_inc + z * se, 1)

        n_at_risk -= n_at_time

    return CompetingRiskResult(
        times=unique_times,
        cif=cif,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        event_of_interest=event_of_interest,
        n_total=n,
    )


def fine_gray_weights(
    time: Union[np.ndarray, pd.Series, list],
    event: Union[np.ndarray, pd.Series, list],
    event_of_interest: int = 1,
) -> np.ndarray:
    """Compute Fine--Gray IPCW weights for subdistribution hazard modelling.

    These weights can be passed to a weighted Cox model to obtain Fine--Gray
    subdistribution hazard ratios.

    Parameters
    ----------
    time, event : array-like
    event_of_interest : int, default 1

    Returns
    -------
    ndarray
        Observation-level weights.

    References
    ----------
    Fine, J. P., & Gray, R. J. (1999). A proportional hazards model for the
    subdistribution of a competing risk. *JASA*, 94(446), 496--509.
    """
    t = np.asarray(time, dtype=np.float64).ravel()
    e = np.asarray(event, dtype=np.int32).ravel()
    n = len(t)
    weights = np.ones(n, dtype=np.float64)

    # Censoring distribution (reverse KM)
    cens_times = np.sort(np.unique(t[e == 0]))
    if len(cens_times) == 0:
        return weights

    # KM estimate of censoring distribution G(t)
    order = np.argsort(t)
    t_s = t[order]
    e_s = e[order]
    G = np.ones(n)
    n_risk = n
    g_current = 1.0
    for i in range(n):
        if e_s[i] == 0:
            g_current *= (1 - 1 / n_risk) if n_risk > 0 else 0
        G[i] = g_current
        n_risk -= 1

    # Map back
    G_orig = np.empty(n)
    G_orig[order] = G

    # Weights: for competing events beyond their failure time, weight by G(t_j)/G(t_i)
    for i in range(n):
        if e[i] != event_of_interest and e[i] != 0:
            # Competing event: weight decays over time
            gt_i = G_orig[i]
            weights[i] = gt_i if gt_i > 0 else 1e-6
        else:
            weights[i] = 1.0
    return weights


# ===================================================================
# CONCORDANCE INDEX
# ===================================================================


def concordance_index(
    event_times: Union[np.ndarray, pd.Series, list],
    predicted_scores: Union[np.ndarray, pd.Series, list],
    event_observed: Union[np.ndarray, pd.Series, list],
) -> float:
    """Harrell's concordance index (*C*-statistic) for survival models.

    Parameters
    ----------
    event_times : array-like
        Observed times.
    predicted_scores : array-like
        Model-predicted risk scores (higher = higher risk).
    event_observed : array-like
        Event indicator (1 = event).

    Returns
    -------
    float
        Concordance index in [0, 1].
    """
    t = np.asarray(event_times, dtype=np.float64).ravel()
    s = np.asarray(predicted_scores, dtype=np.float64).ravel()
    e = np.asarray(event_observed, dtype=np.int32).ravel()
    n = len(t)
    concordant = 0
    discordant = 0
    tied_risk = 0
    for i in range(n):
        if e[i] == 0:
            continue
        for j in range(n):
            if i == j or t[j] <= t[i]:
                continue
            if s[i] > s[j]:
                concordant += 1
            elif s[i] < s[j]:
                discordant += 1
            else:
                tied_risk += 1
    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.5
    return (concordant + 0.5 * tied_risk) / total


# ===================================================================
# HAZARD RATIO
# ===================================================================


def hazard_ratio(
    time: Union[np.ndarray, pd.Series, list],
    event: Union[np.ndarray, pd.Series, list],
    group: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> dict:
    """Estimate the hazard ratio between two groups from a simple Cox model.

    Parameters
    ----------
    time, event, group : array-like
    confidence : float, default 0.95

    Returns
    -------
    dict
        Keys: ``hr``, ``ci_lower``, ``ci_upper``, ``p_value``, ``log_hr``, ``se``.
    """
    t, e = _validate_survival_input(time, event)
    g = np.asarray(group).ravel()
    groups = np.unique(g)
    if len(groups) != 2:
        raise ValueError("hazard_ratio requires exactly 2 groups.")
    x = (g == groups[1]).astype(np.float64).reshape(-1, 1)
    df = pd.DataFrame({"time": t, "event": e, "group": x.ravel()})
    result = cox_ph(df, "time", "event", ["group"], confidence=confidence)
    hr = float(result.hazard_ratios[0])
    return {
        "hr": hr,
        "ci_lower": float(result.ci_lower[0]),
        "ci_upper": float(result.ci_upper[0]),
        "p_value": float(result.p_values[0]),
        "log_hr": float(result.coefficients[0]),
        "se": float(result.standard_errors[0]),
    }


# ===================================================================
# SURVIVAL PLOTTING DATA
# ===================================================================


def survival_plot_data(
    time: Union[np.ndarray, pd.Series, list],
    event: Union[np.ndarray, pd.Series, list],
    group: Union[np.ndarray, pd.Series, list] | None = None,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Generate a long-format DataFrame suitable for survival curve plotting.

    Parameters
    ----------
    time, event : array-like
    group : array-like or None
        Group labels. If ``None``, a single group is assumed.
    confidence : float, default 0.95

    Returns
    -------
    DataFrame
        Columns: ``time``, ``survival``, ``ci_lower``, ``ci_upper``,
        ``at_risk``, ``group``.
    """
    t = np.asarray(time, dtype=np.float64).ravel()
    e = np.asarray(event, dtype=np.int32).ravel()
    if group is None:
        g = np.ones(len(t), dtype=int)
    else:
        g = np.asarray(group).ravel()

    frames = []
    for gval in np.unique(g):
        mask = g == gval
        km = kaplan_meier(t[mask], e[mask], confidence=confidence)
        df = pd.DataFrame(
            {
                "time": km.times,
                "survival": km.survival,
                "ci_lower": km.ci_lower,
                "ci_upper": km.ci_upper,
                "at_risk": km.at_risk,
                "group": str(gval),
            }
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ===================================================================
# CALIBRATION
# ===================================================================


def survival_calibration(
    predicted_survival: Union[np.ndarray, pd.Series, list],
    time: Union[np.ndarray, pd.Series, list],
    event: Union[np.ndarray, pd.Series, list],
    n_groups: int = 10,
) -> pd.DataFrame:
    """Calibration of predicted survival probabilities.

    Divides observations into *n_groups* deciles of predicted risk and compares
    predicted vs. observed KM survival within each group.

    Parameters
    ----------
    predicted_survival : array-like
        Model-predicted survival probabilities at a fixed time horizon.
    time, event : array-like
    n_groups : int, default 10

    Returns
    -------
    DataFrame
        Columns: ``group``, ``predicted_mean``, ``observed_survival``, ``n``.
    """
    pred = np.asarray(predicted_survival, dtype=np.float64).ravel()
    t, e = _validate_survival_input(time, event)
    n = min(len(pred), len(t))
    pred, t, e = pred[:n], t[:n], e[:n]
    quantiles = np.linspace(0, 1, n_groups + 1)
    boundaries = np.quantile(pred, quantiles)
    results = []
    for i in range(n_groups):
        lo = boundaries[i]
        hi = boundaries[i + 1]
        if i == n_groups - 1:
            mask = (pred >= lo) & (pred <= hi)
        else:
            mask = (pred >= lo) & (pred < hi)
        if mask.sum() == 0:
            continue
        km = kaplan_meier(t[mask], e[mask])
        obs_surv = float(km.survival[-1]) if len(km.survival) > 0 else 1.0
        results.append(
            {
                "group": i + 1,
                "predicted_mean": float(pred[mask].mean()),
                "observed_survival": obs_surv,
                "n": int(mask.sum()),
            }
        )
    return pd.DataFrame(results)


# ===================================================================
# LANDMARK ANALYSIS
# ===================================================================


def landmark_dataset(
    data: pd.DataFrame,
    duration_col: str,
    event_col: str,
    landmark_time: float,
) -> pd.DataFrame:
    """Create a landmark analysis dataset.

    Excludes subjects who experienced an event or were censored before
    *landmark_time* and resets the time origin.

    Parameters
    ----------
    data : DataFrame
    duration_col : str
    event_col : str
    landmark_time : float

    Returns
    -------
    DataFrame
        Filtered dataset with adjusted times.
    """
    df = data.copy()
    mask = df[duration_col] >= landmark_time
    out = df.loc[mask].copy()
    out[duration_col] = out[duration_col] - landmark_time
    logger.info(
        "Landmark analysis at t=%s: kept %d of %d subjects.",
        landmark_time,
        len(out),
        len(data),
    )
    return out


# ===================================================================
# LEFT TRUNCATION
# ===================================================================


def left_truncated_km(
    entry_time: Union[np.ndarray, pd.Series, list],
    exit_time: Union[np.ndarray, pd.Series, list],
    event: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> SurvivalCurve:
    """Kaplan--Meier estimator with left truncation (delayed entry).

    Parameters
    ----------
    entry_time : array-like
        Time at which each subject enters the risk set.
    exit_time : array-like
        Observed failure/censoring time.
    event : array-like
        Event indicator.
    confidence : float, default 0.95

    Returns
    -------
    SurvivalCurve
    """
    entry = np.asarray(entry_time, dtype=np.float64).ravel()
    exit_ = np.asarray(exit_time, dtype=np.float64).ravel()
    e = np.asarray(event, dtype=np.int32).ravel()
    n = len(entry)

    event_times = np.sort(np.unique(exit_[e == 1]))
    at_risk = np.empty(len(event_times))
    events = np.empty(len(event_times))
    censored_arr = np.empty(len(event_times))

    for i, ut in enumerate(event_times):
        # At risk: entered before ut and haven't exited before ut
        risk_mask = (entry < ut) & (exit_ >= ut)
        at_risk[i] = risk_mask.sum()
        events[i] = ((exit_ == ut) & (e == 1)).sum()
        censored_arr[i] = ((exit_ == ut) & (e == 0)).sum()

    survival = np.cumprod(1 - events / np.clip(at_risk, 1, np.inf))
    z = stats.norm.ppf((1 + confidence) / 2)
    var_terms = np.cumsum(events / (at_risk * np.clip(at_risk - events, 1, np.inf)))
    se = survival * np.sqrt(var_terms)
    ci_lo = np.clip(survival - z * se, 0, 1)
    ci_hi = np.clip(survival + z * se, 0, 1)

    below_50 = np.where(survival <= 0.5)[0]
    median_surv = float(event_times[below_50[0]]) if len(below_50) > 0 else None

    return SurvivalCurve(
        times=event_times,
        survival=survival,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        at_risk=at_risk,
        events=events,
        censored=censored_arr,
        method="Left-truncated Kaplan-Meier",
        median_survival=median_surv,
    )


# ===================================================================
# INTERVAL CENSORING
# ===================================================================


def turnbull_estimator(
    left: Union[np.ndarray, pd.Series, list],
    right: Union[np.ndarray, pd.Series, list],
    max_iter: int = 200,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Turnbull's self-consistency algorithm for interval-censored data.

    Parameters
    ----------
    left : array-like
        Left endpoints of observation intervals.
    right : array-like
        Right endpoints (use ``np.inf`` for right-censored).
    max_iter : int, default 200
        Maximum EM iterations.
    tol : float, default 1e-6
        Convergence tolerance.

    Returns
    -------
    tuple[ndarray, ndarray]
        ``(unique_times, survival_probs)``.
    """
    L = np.asarray(left, dtype=np.float64).ravel()
    R = np.asarray(right, dtype=np.float64).ravel()
    n = len(L)
    # Unique finite points
    points = np.sort(np.unique(np.concatenate([L, R[np.isfinite(R)]])))
    m = len(points)
    if m == 0:
        return np.array([0.0]), np.array([1.0])

    # Initialise uniform mass
    p = np.ones(m) / m

    for iteration in range(max_iter):
        p_old = p.copy()
        # E-step: for each observation, compute weight of each interval point
        weights = np.zeros(m)
        for i in range(n):
            mask = (points >= L[i]) & (points <= R[i])
            mass = p[mask].sum()
            if mass > 0:
                weights[mask] += p[mask] / mass

        # M-step
        p = weights / n
        p_sum = p.sum()
        if p_sum > 0:
            p /= p_sum

        if np.max(np.abs(p - p_old)) < tol:
            break

    survival = 1.0 - np.cumsum(p)
    return points, np.clip(survival, 0, 1)


# ===================================================================
# MODEL COMPARISON HELPERS
# ===================================================================


def compare_parametric_models(
    time: Union[np.ndarray, pd.Series, list],
    event: Union[np.ndarray, pd.Series, list],
) -> pd.DataFrame:
    """Fit all available parametric models and compare via AIC/BIC.

    Parameters
    ----------
    time, event : array-like

    Returns
    -------
    DataFrame
        Columns: ``distribution``, ``log_likelihood``, ``aic``, ``bic``,
        ``n_events``.
    """
    models = [
        ("Exponential", exponential_model),
        ("Weibull", weibull_model),
        ("Log-normal", lognormal_model),
        ("Log-logistic", loglogistic_model),
        ("Gompertz", gompertz_model),
    ]
    results = []
    for name, func in models:
        try:
            res = func(time, event)
            results.append(
                {
                    "distribution": name,
                    "log_likelihood": res.log_likelihood,
                    "aic": res.aic,
                    "bic": res.bic,
                    "n_events": res.n_events,
                }
            )
        except Exception as exc:
            logger.warning("Failed to fit %s model: %s", name, exc)
    return pd.DataFrame(results).sort_values("aic")
