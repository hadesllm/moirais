"""Model diagnostics and assumption checking for statistical models.

Provides comprehensive diagnostic tools for regression models (linear,
logistic, Cox), including residual analysis, influence diagnostics,
collinearity assessment, specification tests, and goodness-of-fit measures.

All diagnostic functions return structured results compatible with the
MOIRAIS visualization module for publication-ready diagnostic plots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ResidualDiagnostics:
    """Comprehensive residual analysis results."""

    raw_residuals: np.ndarray
    standardized_residuals: np.ndarray
    studentized_residuals: np.ndarray
    deviance_residuals: np.ndarray | None
    pearson_residuals: np.ndarray | None
    fitted_values: np.ndarray
    normality_test: dict[str, float]
    heteroskedasticity_test: dict[str, float]
    autocorrelation_test: dict[str, float]
    n_outliers: int
    outlier_indices: np.ndarray


@dataclass
class InfluenceDiagnostics:
    """Influence and leverage diagnostics."""

    hat_values: np.ndarray
    cooks_distance: np.ndarray
    dffits: np.ndarray
    dfbetas: np.ndarray
    covratio: np.ndarray
    n_influential: int
    influential_indices: np.ndarray
    high_leverage_indices: np.ndarray
    high_cooksd_indices: np.ndarray


@dataclass
class CollinearityDiagnostics:
    """Multicollinearity assessment results."""

    vif: dict[str, float]
    condition_number: float
    condition_indices: np.ndarray
    variance_decomposition: pd.DataFrame
    eigenvalues: np.ndarray
    n_collinear: int
    collinear_pairs: list[tuple[str, str, float]]


@dataclass
class SpecificationTest:
    """Specification/functional form test results."""

    name: str
    statistic: float
    p_value: float
    df: int | None
    conclusion: str


@dataclass
class GoodnessOfFit:
    """Model goodness-of-fit summary."""

    r_squared: float | None
    adj_r_squared: float | None
    pseudo_r_squared: float | None
    aic: float
    bic: float
    log_likelihood: float
    deviance: float | None
    pearson_chi2: float | None
    df_model: int
    df_residual: int
    f_statistic: float | None
    f_pvalue: float | None
    n_obs: int


@dataclass
class DiagnosticReport:
    """Complete diagnostic report for a fitted model."""

    residuals: ResidualDiagnostics
    influence: InfluenceDiagnostics
    collinearity: CollinearityDiagnostics
    goodness_of_fit: GoodnessOfFit
    specification_tests: list[SpecificationTest]
    overall_assessment: str


# ---------------------------------------------------------------------------
# Residual analysis
# ---------------------------------------------------------------------------


def compute_residuals(
    y: np.ndarray,
    y_hat: np.ndarray,
    X: np.ndarray,
    model_type: str = "linear",
) -> ResidualDiagnostics:
    """Compute comprehensive residual diagnostics.

    Parameters
    ----------
    y : array-like
        Observed response.
    y_hat : array-like
        Fitted/predicted values.
    X : array-like
        Design matrix.
    model_type : str
        'linear', 'logistic', or 'poisson'.

    Returns
    -------
    ResidualDiagnostics
    """
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    # Raw residuals.
    raw = y - y_hat

    # Hat matrix diagonal.
    try:
        H = X @ np.linalg.solve(X.T @ X, X.T)
        h = np.diag(H)
    except np.linalg.LinAlgError:
        H = X @ np.linalg.pinv(X.T @ X) @ X.T
        h = np.diag(H)

    h = np.clip(h, 0, 1 - 1e-10)

    # MSE.
    mse = np.sum(raw**2) / max(n - p, 1)

    # Standardized residuals.
    std_res = raw / np.sqrt(max(mse, 1e-10) * (1 - h))

    # Studentized (externally studentized) residuals.
    student_res = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mse_i = np.sum(raw[mask] ** 2) / max(n - p - 1, 1)
        student_res[i] = raw[i] / np.sqrt(max(mse_i * (1 - h[i]), 1e-10))

    # Deviance and Pearson residuals for GLMs.
    deviance_res = None
    pearson_res = None

    if model_type == "logistic":
        y_hat_clip = np.clip(y_hat, 1e-10, 1 - 1e-10)
        # Deviance residuals.
        d = np.where(
            y == 1,
            np.sqrt(-2 * np.log(y_hat_clip)),
            -np.sqrt(-2 * np.log(1 - y_hat_clip)),
        )
        deviance_res = d
        # Pearson residuals.
        pearson_res = (y - y_hat_clip) / np.sqrt(y_hat_clip * (1 - y_hat_clip))

    elif model_type == "poisson":
        y_hat_clip = np.maximum(y_hat, 1e-10)
        deviance_res = np.sign(y - y_hat_clip) * np.sqrt(
            2 * (y * np.log(np.maximum(y / y_hat_clip, 1e-10)) - (y - y_hat_clip))
        )
        pearson_res = (y - y_hat_clip) / np.sqrt(y_hat_clip)

    # Normality test (Shapiro-Wilk on residuals).
    if n <= 5000:
        sw_stat, sw_p = stats.shapiro(raw)
    else:
        sw_stat, sw_p = stats.normaltest(raw)
    normality = {"statistic": float(sw_stat), "p_value": float(sw_p)}

    # Heteroskedasticity (Breusch-Pagan-like: regress squared residuals on X).
    r_sq = raw**2
    r_sq_centered = r_sq - np.mean(r_sq)
    X_centered = X - X.mean(axis=0)
    try:
        ss_reg = np.sum((X_centered @ np.linalg.lstsq(X_centered, r_sq_centered, rcond=None)[0]) ** 2)
        ss_tot = np.sum(r_sq_centered**2)
        bp_stat = n * ss_reg / max(ss_tot, 1e-10)
        bp_p = 1 - stats.chi2.cdf(bp_stat, p - 1)
    except Exception:
        bp_stat, bp_p = float("nan"), float("nan")
    heteroskedasticity = {"statistic": float(bp_stat), "p_value": float(bp_p)}

    # Autocorrelation (Durbin-Watson).
    dw = np.sum(np.diff(raw) ** 2) / max(np.sum(raw**2), 1e-10)
    autocorrelation = {"durbin_watson": float(dw)}

    # Outlier detection (|studentized residual| > 3).
    outlier_mask = np.abs(student_res) > 3
    outlier_indices = np.where(outlier_mask)[0]

    return ResidualDiagnostics(
        raw_residuals=raw,
        standardized_residuals=std_res,
        studentized_residuals=student_res,
        deviance_residuals=deviance_res,
        pearson_residuals=pearson_res,
        fitted_values=y_hat,
        normality_test=normality,
        heteroskedasticity_test=heteroskedasticity,
        autocorrelation_test=autocorrelation,
        n_outliers=int(outlier_mask.sum()),
        outlier_indices=outlier_indices,
    )


# ---------------------------------------------------------------------------
# Influence diagnostics
# ---------------------------------------------------------------------------


def compute_influence(
    y: np.ndarray,
    X: np.ndarray,
    y_hat: np.ndarray | None = None,
) -> InfluenceDiagnostics:
    """Compute influence and leverage diagnostics.

    Parameters
    ----------
    y : array-like
        Observed response.
    X : array-like
        Design matrix.
    y_hat : array-like, optional
        Fitted values.  Computed via OLS if not provided.

    Returns
    -------
    InfluenceDiagnostics
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    # OLS if needed.
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    if y_hat is None:
        y_hat = X @ beta
    residuals = y - y_hat

    # Hat matrix.
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X.T @ X)

    H = X @ XtX_inv @ X.T
    h = np.diag(H)
    h = np.clip(h, 0, 1 - 1e-10)

    mse = np.sum(residuals**2) / max(n - p, 1)

    # Cook's distance.
    cooks_d = (residuals**2 * h) / (p * mse * (1 - h) ** 2 + 1e-10)

    # DFFITS.
    mse_i = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        res_i = residuals[mask]
        mse_i[i] = np.sum(res_i**2) / max(n - p - 1, 1)

    dffits = residuals * np.sqrt(h / ((1 - h) * mse_i + 1e-10)) / np.sqrt(max(mse, 1e-10))

    # DFBETAS.
    dfbetas = np.empty((n, p))
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        try:
            beta_i = np.linalg.lstsq(X[mask], y[mask], rcond=None)[0]
        except Exception:
            beta_i = beta
        se_beta = np.sqrt(np.diag(mse_i[i] * XtX_inv))
        dfbetas[i] = (beta - beta_i) / np.maximum(se_beta, 1e-10)

    # COVRATIO.
    covratio = np.empty(n)
    for i in range(n):
        # COVRATIO_i = det(Var(b_{(i)})) / det(Var(b))
        s2_i = mse_i[i]
        t_i = residuals[i] / np.sqrt(max(s2_i * (1 - h[i]), 1e-10))
        covratio[i] = 1 / ((((n - p - 1 + t_i**2) / (n - p)) ** p) * (1 - h[i]) + 1e-10)

    # Thresholds.
    leverage_threshold = 2 * p / n
    cooksd_threshold = 4 / n

    high_leverage = np.where(h > leverage_threshold)[0]
    high_cooksd = np.where(cooks_d > cooksd_threshold)[0]
    influential = np.union1d(high_leverage, high_cooksd)

    return InfluenceDiagnostics(
        hat_values=h,
        cooks_distance=cooks_d,
        dffits=dffits,
        dfbetas=dfbetas,
        covratio=covratio,
        n_influential=len(influential),
        influential_indices=influential,
        high_leverage_indices=high_leverage,
        high_cooksd_indices=high_cooksd,
    )


# ---------------------------------------------------------------------------
# Collinearity diagnostics
# ---------------------------------------------------------------------------


def compute_vif(X: np.ndarray, column_names: list[str] | None = None) -> dict[str, float]:
    """Compute Variance Inflation Factors.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix (without intercept).
    column_names : list[str], optional
        Column names.

    Returns
    -------
    dict[str, float]
        VIF for each predictor.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    if column_names is None:
        column_names = [f"X{i}" for i in range(p)]

    vifs = {}
    for j in range(p):
        y_j = X[:, j]
        X_j = np.delete(X, j, axis=1)
        X_j = np.column_stack([np.ones(n), X_j])

        try:
            beta = np.linalg.lstsq(X_j, y_j, rcond=None)[0]
            y_hat = X_j @ beta
            ss_res = np.sum((y_j - y_hat) ** 2)
            ss_tot = np.sum((y_j - np.mean(y_j)) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-10)
            vif = 1 / max(1 - r2, 1e-10)
        except Exception:
            vif = float("inf")

        vifs[column_names[j]] = float(vif)

    return vifs


def collinearity_diagnostics(
    X: np.ndarray,
    column_names: list[str] | None = None,
) -> CollinearityDiagnostics:
    """Comprehensive collinearity diagnostics.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix (with or without intercept).
    column_names : list[str], optional
        Column names.

    Returns
    -------
    CollinearityDiagnostics
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    if column_names is None:
        column_names = [f"X{i}" for i in range(p)]

    # VIF.
    vifs = compute_vif(X, column_names)

    # Condition number and indices.
    X_scaled = X / np.sqrt(np.sum(X**2, axis=0) + 1e-10)
    try:
        eigenvalues = np.linalg.svd(X_scaled, compute_uv=False) ** 2
    except np.linalg.LinAlgError:
        eigenvalues = np.ones(p)

    eigenvalues = np.sort(eigenvalues)[::-1]
    condition_number = np.sqrt(eigenvalues[0] / max(eigenvalues[-1], 1e-10))
    condition_indices = np.sqrt(eigenvalues[0] / np.maximum(eigenvalues, 1e-10))

    # Variance decomposition proportions.
    try:
        U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
        phi = Vt.T**2
        denom = np.sum(phi / (S**2 + 1e-10), axis=1, keepdims=True)
        var_decomp = phi / (S**2 + 1e-10) / np.maximum(denom, 1e-10)
        var_decomp_df = pd.DataFrame(
            var_decomp,
            index=column_names,
            columns=[f"comp_{i}" for i in range(p)],
        )
    except Exception:
        var_decomp_df = pd.DataFrame()

    # Collinear pairs (|r| > 0.8).
    corr = np.corrcoef(X.T)
    collinear_pairs = []
    for i in range(p):
        for j in range(i + 1, p):
            if abs(corr[i, j]) > 0.8:
                collinear_pairs.append((column_names[i], column_names[j], float(corr[i, j])))

    n_collinear = sum(1 for v in vifs.values() if v > 10)

    return CollinearityDiagnostics(
        vif=vifs,
        condition_number=float(condition_number),
        condition_indices=condition_indices,
        variance_decomposition=var_decomp_df,
        eigenvalues=eigenvalues,
        n_collinear=n_collinear,
        collinear_pairs=collinear_pairs,
    )


# ---------------------------------------------------------------------------
# Specification tests
# ---------------------------------------------------------------------------


def ramsey_reset_test(
    y: np.ndarray,
    X: np.ndarray,
    powers: tuple = (2, 3),
) -> SpecificationTest:
    """Ramsey RESET test for functional form misspecification.

    Parameters
    ----------
    y : array-like
        Response variable.
    X : array-like
        Design matrix (with intercept).
    powers : tuple
        Powers of fitted values to include.

    Returns
    -------
    SpecificationTest
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    # Restricted model.
    beta_r = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta_r
    ssr_r = np.sum((y - y_hat) ** 2)

    # Unrestricted model: add powers of fitted values.
    X_u = X.copy()
    for power in powers:
        X_u = np.column_stack([X_u, y_hat**power])

    p_u = X_u.shape[1]
    beta_u = np.linalg.lstsq(X_u, y, rcond=None)[0]
    y_hat_u = X_u @ beta_u
    ssr_u = np.sum((y - y_hat_u) ** 2)

    # F-test.
    df1 = p_u - p
    df2 = n - p_u
    f_stat = ((ssr_r - ssr_u) / max(df1, 1)) / (ssr_u / max(df2, 1))
    f_p = 1 - stats.f.cdf(f_stat, df1, df2)

    conclusion = (
        "Reject functional form (p < 0.05): consider nonlinear terms."
        if f_p < 0.05
        else "No evidence of misspecification."
    )

    return SpecificationTest(
        name="RESET",
        statistic=float(f_stat),
        p_value=float(f_p),
        df=df1,
        conclusion=conclusion,
    )


def link_test(
    y: np.ndarray,
    X: np.ndarray,
    model_type: str = "linear",
) -> SpecificationTest:
    """Pregibon link test for GLM specification.

    Parameters
    ----------
    y : array-like
        Response.
    X : array-like
        Design matrix.
    model_type : str
        'linear' or 'logistic'.

    Returns
    -------
    SpecificationTest
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n = len(y)

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta

    # Link test: regress y on y_hat and y_hat^2.
    X_link = np.column_stack([np.ones(n), y_hat, y_hat**2])
    beta_link = np.linalg.lstsq(X_link, y, rcond=None)[0]
    y_link = X_link @ beta_link
    residuals = y - y_link
    mse = np.sum(residuals**2) / max(n - 3, 1)

    # Test significance of y_hat^2 coefficient.
    try:
        XtX_inv = np.linalg.inv(X_link.T @ X_link)
        se = np.sqrt(mse * np.diag(XtX_inv))
        t_stat = beta_link[2] / max(se[2], 1e-10)
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - 3))
    except Exception:
        t_stat, p_val = float("nan"), float("nan")

    conclusion = (
        "Reject link specification (p < 0.05): consider alternative link function."
        if p_val < 0.05
        else "No evidence of link misspecification."
    )

    return SpecificationTest(
        name="link_test",
        statistic=float(t_stat),
        p_value=float(p_val),
        df=n - 3,
        conclusion=conclusion,
    )


def hosmer_lemeshow_test(
    y: np.ndarray,
    y_prob: np.ndarray,
    n_groups: int = 10,
) -> SpecificationTest:
    """Hosmer-Lemeshow goodness-of-fit test for logistic regression.

    Parameters
    ----------
    y : array-like
        Binary response.
    y_prob : array-like
        Predicted probabilities.
    n_groups : int
        Number of decile groups.

    Returns
    -------
    SpecificationTest
    """
    y = np.asarray(y, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y)

    # Create decile groups based on predicted probabilities.
    order = np.argsort(y_prob)
    groups = np.array_split(order, n_groups)

    chi2_stat = 0.0
    for group in groups:
        n_g = len(group)
        if n_g == 0:
            continue
        obs_events = np.sum(y[group])
        exp_events = np.sum(y_prob[group])
        obs_non = n_g - obs_events
        exp_non = n_g - exp_events

        if exp_events > 0:
            chi2_stat += (obs_events - exp_events) ** 2 / exp_events
        if exp_non > 0:
            chi2_stat += (obs_non - exp_non) ** 2 / exp_non

    df = n_groups - 2
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)

    conclusion = (
        "Poor fit (p < 0.05): model does not adequately fit the data."
        if p_value < 0.05
        else "Adequate fit: no evidence of poor calibration."
    )

    return SpecificationTest(
        name="hosmer_lemeshow",
        statistic=float(chi2_stat),
        p_value=float(p_value),
        df=df,
        conclusion=conclusion,
    )


# ---------------------------------------------------------------------------
# Goodness of fit
# ---------------------------------------------------------------------------


def compute_goodness_of_fit(
    y: np.ndarray,
    y_hat: np.ndarray,
    X: np.ndarray,
    model_type: str = "linear",
    log_likelihood: float | None = None,
) -> GoodnessOfFit:
    """Compute comprehensive goodness-of-fit statistics.

    Parameters
    ----------
    y : array-like
        Observed response.
    y_hat : array-like
        Fitted values.
    X : array-like
        Design matrix.
    model_type : str
        'linear', 'logistic', 'poisson'.
    log_likelihood : float, optional
        Log-likelihood of the model (computed if not provided).

    Returns
    -------
    GoodnessOfFit
    """
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    residuals = y - y_hat
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    df_model = p - 1
    df_residual = n - p

    r_squared = None
    adj_r_squared = None
    pseudo_r_squared = None
    f_stat = None
    f_p = None
    deviance = None
    pearson_chi2 = None

    if model_type == "linear":
        r_squared = 1 - ss_res / max(ss_tot, 1e-10)
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / max(df_residual, 1)
        mse_model = (ss_tot - ss_res) / max(df_model, 1)
        mse_res = ss_res / max(df_residual, 1)
        f_stat = mse_model / max(mse_res, 1e-10)
        f_p = 1 - stats.f.cdf(f_stat, df_model, df_residual)

        if log_likelihood is None:
            log_likelihood = -n / 2 * (np.log(2 * np.pi) + np.log(ss_res / n) + 1)

    elif model_type == "logistic":
        y_hat_clip = np.clip(y_hat, 1e-10, 1 - 1e-10)
        if log_likelihood is None:
            log_likelihood = np.sum(y * np.log(y_hat_clip) + (1 - y) * np.log(1 - y_hat_clip))
        # Null log-likelihood.
        p_bar = np.mean(y)
        ll_null = np.sum(y * np.log(p_bar) + (1 - y) * np.log(1 - p_bar))
        pseudo_r_squared = 1 - log_likelihood / min(ll_null, -1e-10)
        deviance = -2 * log_likelihood
        pearson_chi2 = np.sum((y - y_hat_clip) ** 2 / (y_hat_clip * (1 - y_hat_clip) + 1e-10))

    elif model_type == "poisson":
        y_hat_clip = np.maximum(y_hat, 1e-10)
        if log_likelihood is None:
            log_likelihood = np.sum(y * np.log(y_hat_clip) - y_hat_clip)
        ll_null = np.sum(y * np.log(np.mean(y)) - np.mean(y))
        pseudo_r_squared = 1 - log_likelihood / min(ll_null, -1e-10)
        deviance = 2 * np.sum(y * np.log(np.maximum(y / y_hat_clip, 1e-10)) - (y - y_hat_clip))
        pearson_chi2 = np.sum((y - y_hat_clip) ** 2 / y_hat_clip)

    # AIC and BIC.
    aic = -2 * log_likelihood + 2 * p
    bic = -2 * log_likelihood + np.log(n) * p

    return GoodnessOfFit(
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        pseudo_r_squared=pseudo_r_squared,
        aic=float(aic),
        bic=float(bic),
        log_likelihood=float(log_likelihood),
        deviance=deviance,
        pearson_chi2=pearson_chi2,
        df_model=df_model,
        df_residual=df_residual,
        f_statistic=float(f_stat) if f_stat is not None else None,
        f_pvalue=float(f_p) if f_p is not None else None,
        n_obs=n,
    )


# ---------------------------------------------------------------------------
# Proportional hazards diagnostics
# ---------------------------------------------------------------------------


def ph_assumption_test(
    survival_times: np.ndarray,
    event_indicator: np.ndarray,
    covariates: np.ndarray,
    covariate_names: list[str] | None = None,
) -> list[SpecificationTest]:
    """Test the proportional hazards assumption using Schoenfeld residuals.

    Parameters
    ----------
    survival_times : array-like
        Event or censoring times.
    event_indicator : array-like
        1 if event, 0 if censored.
    covariates : array-like
        Covariate matrix.
    covariate_names : list[str], optional
        Names for each covariate.

    Returns
    -------
    list[SpecificationTest]
        One test result per covariate.
    """
    times = np.asarray(survival_times, dtype=float)
    events = np.asarray(event_indicator, dtype=int)
    X = np.asarray(covariates, dtype=float)
    n, p = X.shape

    if covariate_names is None:
        covariate_names = [f"X{i}" for i in range(p)]

    results = []

    # Simple test: correlate each covariate with rank-transformed event times.
    event_mask = events == 1
    event_times = times[event_mask]
    event_X = X[event_mask]

    ranked_times = stats.rankdata(event_times)

    for j in range(p):
        rho, p_val = stats.spearmanr(ranked_times, event_X[:, j])

        conclusion = (
            f"PH violation for {covariate_names[j]} (p < 0.05)."
            if p_val < 0.05
            else f"PH assumption holds for {covariate_names[j]}."
        )

        results.append(
            SpecificationTest(
                name=f"ph_test:{covariate_names[j]}",
                statistic=float(rho),
                p_value=float(p_val),
                df=len(event_times) - 2,
                conclusion=conclusion,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------


def likelihood_ratio_test(
    ll_restricted: float,
    ll_full: float,
    df_diff: int,
) -> SpecificationTest:
    """Likelihood ratio test for nested models.

    Parameters
    ----------
    ll_restricted : float
        Log-likelihood of the restricted model.
    ll_full : float
        Log-likelihood of the full model.
    df_diff : int
        Difference in degrees of freedom.

    Returns
    -------
    SpecificationTest
    """
    lr_stat = -2 * (ll_restricted - ll_full)
    p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)

    conclusion = (
        "Full model significantly improves fit (p < 0.05)."
        if p_value < 0.05
        else "No significant improvement from full model."
    )

    return SpecificationTest(
        name="likelihood_ratio",
        statistic=float(lr_stat),
        p_value=float(p_value),
        df=df_diff,
        conclusion=conclusion,
    )


def wald_test(
    estimates: np.ndarray,
    vcov: np.ndarray,
    R: np.ndarray | None = None,
    r: np.ndarray | None = None,
) -> SpecificationTest:
    """Wald test for linear restrictions on parameters.

    Tests H0: R @ beta = r.

    Parameters
    ----------
    estimates : array-like
        Parameter estimates.
    vcov : array-like
        Variance-covariance matrix.
    R : array-like, optional
        Restriction matrix.  Default: identity (test all params = 0).
    r : array-like, optional
        Restriction vector.  Default: zeros.

    Returns
    -------
    SpecificationTest
    """
    beta = np.asarray(estimates, dtype=float)
    V = np.asarray(vcov, dtype=float)
    p = len(beta)

    if R is None:
        R = np.eye(p)
    else:
        R = np.asarray(R, dtype=float)

    if r is None:
        r = np.zeros(R.shape[0])
    else:
        r = np.asarray(r, dtype=float)

    q = R.shape[0]  # Number of restrictions.
    diff = R @ beta - r

    try:
        meat = R @ V @ R.T
        w_stat = float(diff @ np.linalg.solve(meat, diff))
    except np.linalg.LinAlgError:
        w_stat = float(diff @ np.linalg.pinv(meat) @ diff)

    p_value = 1 - stats.chi2.cdf(w_stat, q)

    conclusion = "Reject null hypothesis (p < 0.05)." if p_value < 0.05 else "Cannot reject null hypothesis."

    return SpecificationTest(
        name="wald",
        statistic=w_stat,
        p_value=float(p_value),
        df=q,
        conclusion=conclusion,
    )


def score_test(
    score_vector: np.ndarray,
    information_matrix: np.ndarray,
) -> SpecificationTest:
    """Score (Lagrange multiplier) test.

    Parameters
    ----------
    score_vector : array-like
        Score vector evaluated under H0.
    information_matrix : array-like
        Expected or observed information matrix under H0.

    Returns
    -------
    SpecificationTest
    """
    U = np.asarray(score_vector, dtype=float)
    I = np.asarray(information_matrix, dtype=float)
    q = len(U)

    try:
        s_stat = float(U @ np.linalg.solve(I, U))
    except np.linalg.LinAlgError:
        s_stat = float(U @ np.linalg.pinv(I) @ U)

    p_value = 1 - stats.chi2.cdf(s_stat, q)

    return SpecificationTest(
        name="score",
        statistic=s_stat,
        p_value=float(p_value),
        df=q,
        conclusion="Reject H0." if p_value < 0.05 else "Cannot reject H0.",
    )


# ---------------------------------------------------------------------------
# Comprehensive diagnostic report
# ---------------------------------------------------------------------------


def full_diagnostics(
    y: np.ndarray,
    X: np.ndarray,
    y_hat: np.ndarray | None = None,
    model_type: str = "linear",
    column_names: list[str] | None = None,
) -> DiagnosticReport:
    """Run all diagnostic tests and return a comprehensive report.

    Parameters
    ----------
    y : array-like
        Observed response.
    X : array-like
        Design matrix.
    y_hat : array-like, optional
        Fitted values.
    model_type : str
        'linear', 'logistic', 'poisson'.
    column_names : list[str], optional
        Column names for X.

    Returns
    -------
    DiagnosticReport
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)

    if y_hat is None:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta

    residuals = compute_residuals(y, y_hat, X, model_type)
    influence = compute_influence(y, X, y_hat)
    collinearity = collinearity_diagnostics(X, column_names)
    gof = compute_goodness_of_fit(y, y_hat, X, model_type)

    spec_tests = []
    if model_type == "linear":
        spec_tests.append(ramsey_reset_test(y, X))
    spec_tests.append(link_test(y, X, model_type))
    if model_type == "logistic":
        spec_tests.append(hosmer_lemeshow_test(y, y_hat))

    # Overall assessment.
    issues = []
    if residuals.normality_test["p_value"] < 0.05:
        issues.append("non-normal residuals")
    if residuals.heteroskedasticity_test["p_value"] < 0.05:
        issues.append("heteroskedasticity")
    if residuals.n_outliers > 0:
        issues.append(f"{residuals.n_outliers} outlier(s)")
    if influence.n_influential > 0:
        issues.append(f"{influence.n_influential} influential point(s)")
    if collinearity.n_collinear > 0:
        issues.append(f"{collinearity.n_collinear} collinear variable(s)")

    if issues:
        assessment = "Issues detected: " + "; ".join(issues)
    else:
        assessment = "No major diagnostic issues detected."

    return DiagnosticReport(
        residuals=residuals,
        influence=influence,
        collinearity=collinearity,
        goodness_of_fit=gof,
        specification_tests=spec_tests,
        overall_assessment=assessment,
    )
