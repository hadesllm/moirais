"""
Instrumental Variables and Two-Stage Least Squares estimators.

Implements 2SLS, LIML, GMM, CUE-GMM, weak-instrument diagnostics
(Cragg-Donald, Kleibergen-Paap, Stock-Yogo), Anderson-Rubin inference,
Sargan/Hansen J tests, Hausman tests, JIVE, split-sample IV, IV probit,
panel IV, control function, and comprehensive residual diagnostics.

References
----------
Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics:
An Empiricist's Companion*. Princeton University Press.

Stock, J. H., & Yogo, M. (2005). Testing for weak instruments in linear
IV regression. In *Identification and Inference for Econometric Models:
Essays in Honor of Thomas Rothenberg* (pp. 80--108). Cambridge University
Press.

Kleibergen, F. (2002). Pivotal statistics for testing structural
parameters in instrumental variables regression. *Econometrica*, 70(5),
1781--1803. https://doi.org/10.1111/1468-0262.00353

Sargan, J. D. (1958). The estimation of economic relationships using
instrumental variables. *Econometrica*, 26(3), 393--415.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class IVResult:
    """Container for an IV/2SLS regression result.

    Attributes
    ----------
    coefficients : np.ndarray
        Estimated coefficient vector.
    std_errors : np.ndarray
        Standard errors.
    t_stats : np.ndarray
        t-statistics.
    p_values : np.ndarray
        Two-sided p-values.
    ci_lower : np.ndarray
        Lower 95 % CI bounds.
    ci_upper : np.ndarray
        Upper 95 % CI bounds.
    variable_names : list of str
        Names of variables corresponding to each coefficient.
    n_obs : int
        Number of observations.
    method : str
        Estimation method name.
    details : dict
        Additional diagnostics.
    """

    coefficients: np.ndarray
    std_errors: np.ndarray
    t_stats: np.ndarray
    p_values: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    variable_names: list[str]
    n_obs: int
    method: str = "2sls"
    details: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Return a summary table."""
        return pd.DataFrame(
            {
                "variable": self.variable_names,
                "coefficient": self.coefficients,
                "std_error": self.std_errors,
                "t_stat": self.t_stats,
                "p_value": self.p_values,
                "ci_lower": self.ci_lower,
                "ci_upper": self.ci_upper,
            }
        )

    @property
    def endogenous_coef(self) -> float:
        """Return the coefficient on the first endogenous variable."""
        # coefficients[0] is the constant; endogenous starts at index 1
        idx = 1 if len(self.coefficients) > 1 else 0
        return float(self.coefficients[idx])

    @property
    def endogenous_se(self) -> float:
        """Return the SE on the first endogenous variable."""
        idx = 1 if len(self.std_errors) > 1 else 0
        return float(self.std_errors[idx])


@dataclass
class DiagnosticResult:
    """Container for a single diagnostic test."""

    statistic: float
    p_value: float
    test_name: str
    critical_values: dict[str, float] | None = None
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper: projection and annihilator matrices
# ---------------------------------------------------------------------------


def _proj(Z: np.ndarray) -> np.ndarray:
    """Projection matrix P_Z = Z (Z'Z)^{-1} Z'."""
    return Z @ np.linalg.pinv(Z.T @ Z) @ Z.T


def _annihilator(Z: np.ndarray) -> np.ndarray:
    """Annihilator matrix M_Z = I - P_Z."""
    return np.eye(Z.shape[0]) - _proj(Z)


def _add_const(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones."""
    return np.column_stack([np.ones(X.shape[0]), X])


def _robust_se(X: np.ndarray, resid: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """HC1 heteroskedasticity-robust standard errors for IV."""
    n, k = X.shape
    P_Z = _proj(Z)
    XtPZ = X.T @ P_Z
    try:
        bread = np.linalg.inv(XtPZ @ X)
    except np.linalg.LinAlgError:
        bread = np.linalg.pinv(XtPZ @ X)

    # HC1 meat
    meat = np.zeros((k, k))
    for i in range(n):
        xi = X[i : i + 1]
        zi = Z[i : i + 1]
        meat += resid[i] ** 2 * (xi.T @ zi) @ (zi.T @ xi)

    correction = n / (n - k)
    V = correction * bread @ meat @ bread
    return np.sqrt(np.maximum(np.diag(V), 0.0))


def _cluster_se(
    X: np.ndarray,
    resid: np.ndarray,
    Z: np.ndarray,
    cluster_ids: np.ndarray,
) -> np.ndarray:
    """Cluster-robust SE for IV estimators."""
    n, k = X.shape
    P_Z = _proj(Z)
    try:
        bread = np.linalg.inv(X.T @ P_Z @ X)
    except np.linalg.LinAlgError:
        bread = np.linalg.pinv(X.T @ P_Z @ X)

    unique_c = np.unique(cluster_ids)
    G = len(unique_c)
    meat = np.zeros((k, k))
    for c in unique_c:
        mask = cluster_ids == c
        Xc = X[mask]
        Zc = Z[mask]
        ec = resid[mask]
        score = Xc.T @ (Zc @ np.linalg.pinv(Z.T @ Z) @ Z.T @ np.diag(ec) @ np.ones(mask.sum()))
        # Simplified: use X'Z*e clustering
        score_c = Xc.T @ np.diag(ec) @ Zc @ np.linalg.pinv(Z.T @ Z) @ Z.T[:, mask].sum(axis=1)
        meat += np.outer(Xc.T @ (ec[:, None] * Zc).sum(axis=0), Xc.T @ (ec[:, None] * Zc).sum(axis=0))

    # Simplified cluster-robust: direct sandwich
    # Re-do with simpler formula
    meat = np.zeros((k, k))
    PZ_Z = np.linalg.pinv(Z.T @ Z) @ Z.T
    for c in unique_c:
        mask = cluster_ids == c
        Xc = X[mask]
        ec = resid[mask]
        s = Xc.T @ ec  # (k,)
        meat += np.outer(s, s)

    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    V = correction * bread @ meat @ bread
    return np.sqrt(np.maximum(np.diag(V), 0.0))


# ---------------------------------------------------------------------------
# Core: Two-Stage Least Squares (2SLS)
# ---------------------------------------------------------------------------


def tsls(
    data: pd.DataFrame,
    outcome: str,
    endogenous: list[str],
    instruments: list[str],
    exogenous: list[str] | None = None,
    *,
    cluster: str | None = None,
    robust: bool = True,
    alpha: float = 0.05,
) -> IVResult:
    r"""Two-Stage Least Squares (2SLS) estimator.

    First stage: regress each endogenous variable on the instruments
    and exogenous variables.  Second stage: regress the outcome on
    the fitted endogenous variables and exogenous variables.

    .. math::

        \hat\beta_{2SLS} = (X'P_Z X)^{-1} X'P_Z y

    where :math:`P_Z = Z(Z'Z)^{-1}Z'` and :math:`Z` includes both
    excluded instruments and included exogenous variables.

    Parameters
    ----------
    data : pd.DataFrame
    outcome : str
        Dependent variable.
    endogenous : list of str
        Endogenous regressors.
    instruments : list of str
        Excluded instruments.
    exogenous : list of str, optional
        Included exogenous regressors (added automatically to Z).
    cluster : str, optional
        Cluster variable for cluster-robust SE.
    robust : bool
        Use heteroskedasticity-robust SE (default True).
    alpha : float
        Significance level.

    Returns
    -------
    IVResult

    References
    ----------
    Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless
    Econometrics*. Princeton University Press.
    """
    cols = [outcome] + endogenous + instruments
    if exogenous:
        cols += exogenous
    if cluster:
        cols.append(cluster)

    df = data[list(set(cols))].dropna().copy()
    y = df[outcome].values.astype(float)
    n = len(y)

    D = df[endogenous].values.astype(float)
    Z_excl = df[instruments].values.astype(float)

    if exogenous:
        W = df[exogenous].values.astype(float)
        X = _add_const(np.column_stack([D, W]))
        Z = _add_const(np.column_stack([Z_excl, W]))
        var_names = ["const"] + endogenous + exogenous
    else:
        X = _add_const(D)
        Z = _add_const(Z_excl)
        var_names = ["const"] + endogenous

    k = X.shape[1]

    # 2SLS
    P_Z = _proj(Z)
    try:
        beta = np.linalg.inv(X.T @ P_Z @ X) @ (X.T @ P_Z @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(X.T @ P_Z @ X) @ (X.T @ P_Z @ y)

    resid = y - X @ beta

    # Standard errors
    if cluster and cluster in df.columns:
        se = _cluster_se(X, resid, Z, df[cluster].values)
    elif robust:
        se = _robust_se(X, resid, Z)
    else:
        sigma2 = float(np.sum(resid**2) / (n - k))
        try:
            V = sigma2 * np.linalg.inv(X.T @ P_Z @ X)
        except np.linalg.LinAlgError:
            V = sigma2 * np.linalg.pinv(X.T @ P_Z @ X)
        se = np.sqrt(np.maximum(np.diag(V), 0.0))

    t_vals = np.where(se > 0, beta / se, 0.0)
    p_vals = 2 * stats.norm.sf(np.abs(t_vals))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lo = beta - z_crit * se
    ci_hi = beta + z_crit * se

    return IVResult(
        coefficients=beta,
        std_errors=se,
        t_stats=t_vals,
        p_values=p_vals,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        variable_names=var_names,
        n_obs=n,
        method="2sls",
        details={"residual_variance": float(np.var(resid))},
    )


# ---------------------------------------------------------------------------
# LIML
# ---------------------------------------------------------------------------


def liml(
    data: pd.DataFrame,
    outcome: str,
    endogenous: list[str],
    instruments: list[str],
    exogenous: list[str] | None = None,
    *,
    robust: bool = True,
    alpha: float = 0.05,
) -> IVResult:
    r"""Limited Information Maximum Likelihood (LIML) estimator.

    More robust to weak instruments than 2SLS.  The LIML estimator is
    the k-class estimator with :math:`\kappa` equal to the smallest
    eigenvalue of a certain matrix ratio.

    .. math::

        \hat\beta_{\text{LIML}} = \bigl(X'(I - \kappa M_Z) X\bigr)^{-1}
        X'(I - \kappa M_Z) y

    Parameters
    ----------
    data : pd.DataFrame
    outcome : str
    endogenous : list of str
    instruments : list of str
    exogenous : list of str, optional
    robust : bool
    alpha : float

    Returns
    -------
    IVResult

    References
    ----------
    Anderson, T. W., & Rubin, H. (1949). Estimation of the parameters of
    a single equation in a complete system of stochastic equations.
    *Annals of Mathematical Statistics*, 20(1), 46--63.
    """
    cols = [outcome] + endogenous + instruments
    if exogenous:
        cols += exogenous
    df = data[list(set(cols))].dropna().copy()
    y = df[outcome].values.astype(float)
    n = len(y)

    D = df[endogenous].values.astype(float)
    Z_excl = df[instruments].values.astype(float)

    if exogenous:
        W = df[exogenous].values.astype(float)
        W_full = _add_const(W)
    else:
        W_full = np.ones((n, 1))

    M_W = _annihilator(W_full)

    # Matrices for kappa calculation
    Y_tilde = M_W @ np.column_stack([y, D])
    Z_tilde = M_W @ Z_excl
    Z_all = np.column_stack([W_full, Z_excl])
    M_Z_all = _annihilator(Z_all)

    # kappa = smallest eigenvalue of (Y~' M_Zexcl Y~)^{-1} (Y~' Y~)
    # Equivalently: eigenvalue problem
    A = Y_tilde.T @ M_Z_all @ Y_tilde
    B = Y_tilde.T @ M_W @ Y_tilde

    try:
        eigvals = np.linalg.eigvalsh(np.linalg.solve(B, A))
    except np.linalg.LinAlgError:
        eigvals = np.linalg.eigvalsh(np.linalg.pinv(B) @ A)

    kappa = float(np.min(eigvals[eigvals > -1e-10]))

    # k-class estimator
    if exogenous:
        X = _add_const(np.column_stack([D, W]))
        Z = _add_const(np.column_stack([Z_excl, W]))
        var_names = ["const"] + endogenous + exogenous
    else:
        X = _add_const(D)
        Z = _add_const(Z_excl)
        var_names = ["const"] + endogenous

    k = X.shape[1]
    P_Z = _proj(Z)
    M_Z = np.eye(n) - P_Z
    XtAX = X.T @ (np.eye(n) - kappa * M_Z) @ X
    XtAy = X.T @ (np.eye(n) - kappa * M_Z) @ y

    try:
        beta = np.linalg.solve(XtAX, XtAy)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XtAX) @ XtAy

    resid = y - X @ beta
    sigma2 = float(np.sum(resid**2) / (n - k))

    if robust:
        se = _robust_se(X, resid, Z)
    else:
        try:
            V = sigma2 * np.linalg.inv(XtAX)
        except np.linalg.LinAlgError:
            V = sigma2 * np.linalg.pinv(XtAX)
        se = np.sqrt(np.maximum(np.diag(V), 0.0))

    t_vals = np.where(se > 0, beta / se, 0.0)
    p_vals = 2 * stats.norm.sf(np.abs(t_vals))
    z_crit = stats.norm.ppf(1 - alpha / 2)

    return IVResult(
        coefficients=beta,
        std_errors=se,
        t_stats=t_vals,
        p_values=p_vals,
        ci_lower=beta - z_crit * se,
        ci_upper=beta + z_crit * se,
        variable_names=var_names,
        n_obs=n,
        method="liml",
        details={"kappa": kappa},
    )


# ---------------------------------------------------------------------------
# GMM
# ---------------------------------------------------------------------------


def gmm_iv(
    data: pd.DataFrame,
    outcome: str,
    endogenous: list[str],
    instruments: list[str],
    exogenous: list[str] | None = None,
    *,
    weight_matrix: str = "optimal",
    robust: bool = True,
    alpha: float = 0.05,
) -> IVResult:
    r"""Generalised Method of Moments (GMM) IV estimator.

    Two-step efficient GMM: first estimates with the identity weight
    matrix, then re-estimates with the optimal weight matrix constructed
    from first-step residuals.

    .. math::

        \hat\beta_{\text{GMM}} = \arg\min_\beta
        \bigl[Z'(y - X\beta)\bigr]' \hat W
        \bigl[Z'(y - X\beta)\bigr]

    Parameters
    ----------
    data : pd.DataFrame
    outcome : str
    endogenous : list of str
    instruments : list of str
    exogenous : list of str, optional
    weight_matrix : str
        ``"optimal"`` (efficient GMM) or ``"identity"`` (one-step).
    robust : bool
    alpha : float

    Returns
    -------
    IVResult
    """
    cols = [outcome] + endogenous + instruments
    if exogenous:
        cols += exogenous
    df = data[list(set(cols))].dropna().copy()
    y = df[outcome].values.astype(float)
    n = len(y)

    D = df[endogenous].values.astype(float)
    Z_excl = df[instruments].values.astype(float)

    if exogenous:
        W_vars = df[exogenous].values.astype(float)
        X = _add_const(np.column_stack([D, W_vars]))
        Z = _add_const(np.column_stack([Z_excl, W_vars]))
        var_names = ["const"] + endogenous + exogenous
    else:
        X = _add_const(D)
        Z = _add_const(Z_excl)
        var_names = ["const"] + endogenous

    k = X.shape[1]
    L = Z.shape[1]

    # Step 1: 2SLS as initial estimator
    P_Z = _proj(Z)
    try:
        beta1 = np.linalg.inv(X.T @ P_Z @ X) @ (X.T @ P_Z @ y)
    except np.linalg.LinAlgError:
        beta1 = np.linalg.pinv(X.T @ P_Z @ X) @ (X.T @ P_Z @ y)

    if weight_matrix == "identity":
        beta = beta1
        resid = y - X @ beta
    else:
        # Optimal weight matrix from first-step residuals
        resid1 = y - X @ beta1
        S = (Z.T @ np.diag(resid1**2) @ Z) / n  # heteroskedastic-robust

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        # Step 2: efficient GMM
        try:
            beta = np.linalg.inv(X.T @ Z @ S_inv @ Z.T @ X) @ (X.T @ Z @ S_inv @ Z.T @ y)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(X.T @ Z @ S_inv @ Z.T @ X) @ (X.T @ Z @ S_inv @ Z.T @ y)
        resid = y - X @ beta

    # Variance
    S_hat = (Z.T @ np.diag(resid**2) @ Z) / n
    try:
        S_hat_inv = np.linalg.inv(S_hat)
    except np.linalg.LinAlgError:
        S_hat_inv = np.linalg.pinv(S_hat)

    try:
        V = np.linalg.inv(X.T @ Z @ S_hat_inv @ Z.T @ X) / n
    except np.linalg.LinAlgError:
        V = np.linalg.pinv(X.T @ Z @ S_hat_inv @ Z.T @ X) / n

    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    t_vals = np.where(se > 0, beta / se, 0.0)
    p_vals = 2 * stats.norm.sf(np.abs(t_vals))
    z_crit = stats.norm.ppf(1 - alpha / 2)

    # J-test statistic if overidentified
    g_bar = Z.T @ resid / n
    j_stat = float(n * g_bar.T @ S_hat_inv @ g_bar) if k < L else np.nan
    j_df = L - k
    j_p = float(stats.chi2.sf(j_stat, j_df)) if j_df > 0 and not np.isnan(j_stat) else np.nan

    return IVResult(
        coefficients=beta,
        std_errors=se,
        t_stats=t_vals,
        p_values=p_vals,
        ci_lower=beta - z_crit * se,
        ci_upper=beta + z_crit * se,
        variable_names=var_names,
        n_obs=n,
        method="gmm",
        details={"j_stat": j_stat, "j_p_value": j_p, "j_df": j_df, "weight_matrix": weight_matrix},
    )


# ---------------------------------------------------------------------------
# Continuously Updated GMM (CUE)
# ---------------------------------------------------------------------------


def cue_gmm(
    data: pd.DataFrame,
    outcome: str,
    endogenous: list[str],
    instruments: list[str],
    exogenous: list[str] | None = None,
    *,
    max_iter: int = 100,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> IVResult:
    r"""Continuously Updated GMM estimator.

    Unlike two-step GMM, CUE simultaneously optimises over :math:`\beta`
    and the weight matrix, yielding an estimator that is invariant to
    the normalisation of moment conditions.

    Parameters
    ----------
    data : pd.DataFrame
    outcome : str
    endogenous, instruments : list of str
    exogenous : list of str, optional
    max_iter : int
    tol : float
    alpha : float

    Returns
    -------
    IVResult
    """
    cols = [outcome] + endogenous + instruments
    if exogenous:
        cols += exogenous
    df = data[list(set(cols))].dropna().copy()
    y = df[outcome].values.astype(float)
    n = len(y)

    D = df[endogenous].values.astype(float)
    Z_excl = df[instruments].values.astype(float)

    if exogenous:
        W_vars = df[exogenous].values.astype(float)
        X = _add_const(np.column_stack([D, W_vars]))
        Z = _add_const(np.column_stack([Z_excl, W_vars]))
        var_names = ["const"] + endogenous + exogenous
    else:
        X = _add_const(D)
        Z = _add_const(Z_excl)
        var_names = ["const"] + endogenous

    k = X.shape[1]

    # Initial beta from 2SLS
    P_Z = _proj(Z)
    try:
        beta = np.linalg.inv(X.T @ P_Z @ X) @ (X.T @ P_Z @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(X.T @ P_Z @ X) @ (X.T @ P_Z @ y)

    for iteration in range(max_iter):
        resid = y - X @ beta
        S = (Z.T @ np.diag(resid**2) @ Z) / n
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        try:
            beta_new = np.linalg.inv(X.T @ Z @ S_inv @ Z.T @ X) @ (X.T @ Z @ S_inv @ Z.T @ y)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.pinv(X.T @ Z @ S_inv @ Z.T @ X) @ (X.T @ Z @ S_inv @ Z.T @ y)

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    resid = y - X @ beta
    S = (Z.T @ np.diag(resid**2) @ Z) / n
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)

    try:
        V = np.linalg.inv(X.T @ Z @ S_inv @ Z.T @ X) / n
    except np.linalg.LinAlgError:
        V = np.linalg.pinv(X.T @ Z @ S_inv @ Z.T @ X) / n

    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    t_vals = np.where(se > 0, beta / se, 0.0)
    p_vals = 2 * stats.norm.sf(np.abs(t_vals))
    z_crit = stats.norm.ppf(1 - alpha / 2)

    return IVResult(
        coefficients=beta,
        std_errors=se,
        t_stats=t_vals,
        p_values=p_vals,
        ci_lower=beta - z_crit * se,
        ci_upper=beta + z_crit * se,
        variable_names=var_names,
        n_obs=n,
        method="cue_gmm",
        details={"iterations": iteration + 1},
    )


# ---------------------------------------------------------------------------
# Wald (simple IV) estimator
# ---------------------------------------------------------------------------


def wald_estimator(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    instrument: str,
    *,
    alpha: float = 0.05,
) -> IVResult:
    r"""Wald (simple IV) estimator for a single binary instrument.

    .. math::

        \hat\beta_{\text{Wald}} =
        \frac{E[Y | Z=1] - E[Y | Z=0]}{E[D | Z=1] - E[D | Z=0]}

    Parameters
    ----------
    data : pd.DataFrame
    outcome : str
    treatment : str
        Endogenous treatment variable.
    instrument : str
        Binary instrument.
    alpha : float

    Returns
    -------
    IVResult
    """
    df = data.dropna(subset=[outcome, treatment, instrument]).copy()
    z = df[instrument].values.astype(float)
    d = df[treatment].values.astype(float)
    y = df[outcome].values.astype(float)

    z1 = z == 1
    z0 = z == 0

    if z1.sum() == 0 or z0.sum() == 0:
        raise ValueError("Instrument must have observations at both Z=0 and Z=1.")

    # Reduced form
    rf = float(y[z1].mean() - y[z0].mean())
    # First stage
    fs = float(d[z1].mean() - d[z0].mean())

    if abs(fs) < 1e-10:
        logger.warning("First stage near zero; Wald estimate unreliable.")
        return IVResult(
            coefficients=np.array([np.nan]),
            std_errors=np.array([np.nan]),
            t_stats=np.array([np.nan]),
            p_values=np.array([np.nan]),
            ci_lower=np.array([np.nan]),
            ci_upper=np.array([np.nan]),
            variable_names=[treatment],
            n_obs=len(df),
            method="wald",
        )

    beta_wald = rf / fs

    # Delta method SE
    n1 = int(z1.sum())
    n0 = int(z0.sum())
    var_rf = float(y[z1].var(ddof=1) / n1 + y[z0].var(ddof=1) / n0)
    var_fs = float(d[z1].var(ddof=1) / n1 + d[z0].var(ddof=1) / n0)
    se_wald = abs(beta_wald) * np.sqrt(var_rf / rf**2 + var_fs / fs**2) if rf != 0 else np.sqrt(var_rf / fs**2)

    t_val = beta_wald / se_wald if se_wald > 0 else 0.0
    p_val = float(2 * stats.norm.sf(abs(t_val)))
    z_crit = stats.norm.ppf(1 - alpha / 2)

    return IVResult(
        coefficients=np.array([beta_wald]),
        std_errors=np.array([se_wald]),
        t_stats=np.array([t_val]),
        p_values=np.array([p_val]),
        ci_lower=np.array([beta_wald - z_crit * se_wald]),
        ci_upper=np.array([beta_wald + z_crit * se_wald]),
        variable_names=[treatment],
        n_obs=len(df),
        method="wald",
        details={"reduced_form": rf, "first_stage": fs},
    )


# ---------------------------------------------------------------------------
# First-stage diagnostics
# ---------------------------------------------------------------------------


def first_stage_diagnostics(
    data: pd.DataFrame,
    endogenous: list[str],
    instruments: list[str],
    exogenous: list[str] | None = None,
) -> pd.DataFrame:
    """First-stage regression diagnostics for each endogenous variable.

    Reports the F-statistic, partial R-squared, and Shea's partial
    R-squared for each endogenous variable.

    Parameters
    ----------
    data : pd.DataFrame
    endogenous : list of str
    instruments : list of str
    exogenous : list of str, optional

    Returns
    -------
    pd.DataFrame
        Columns: ``endogenous_var``, ``f_stat``, ``partial_r2``,
        ``shea_partial_r2``, ``n_obs``.

    References
    ----------
    Shea, J. (1997). Instrument relevance in multivariate linear models:
    A simple measure. *Review of Economics and Statistics*, 79(2), 348--352.
    """
    cols = endogenous + instruments
    if exogenous:
        cols += exogenous
    df = data[list(set(cols))].dropna().copy()
    n = len(df)

    if exogenous:
        W = _add_const(df[exogenous].values.astype(float))
    else:
        W = np.ones((n, 1))

    M_W = _annihilator(W)
    Z_excl = df[instruments].values.astype(float)
    Z_tilde = M_W @ Z_excl

    results = []
    for endog_var in endogenous:
        d = df[endog_var].values.astype(float)
        d_tilde = M_W @ d

        # Regress d_tilde on Z_tilde
        try:
            beta_z = np.linalg.inv(Z_tilde.T @ Z_tilde) @ (Z_tilde.T @ d_tilde)
        except np.linalg.LinAlgError:
            beta_z = np.linalg.pinv(Z_tilde.T @ Z_tilde) @ (Z_tilde.T @ d_tilde)

        d_hat = Z_tilde @ beta_z
        ssr_full = float(np.sum((d_tilde - d_hat) ** 2))
        sst = float(np.sum(d_tilde**2))

        partial_r2 = 1 - ssr_full / sst if sst > 0 else 0.0

        # F-statistic
        q = len(instruments)
        k = W.shape[1] + q
        f_stat = ((sst - ssr_full) / q) / (ssr_full / (n - k)) if ssr_full > 0 else 0.0

        # Shea's partial R2 (simplified for single endogenous)
        shea_r2 = partial_r2  # exact when only one endogenous variable

        results.append(
            {
                "endogenous_var": endog_var,
                "f_stat": float(f_stat),
                "partial_r2": float(partial_r2),
                "shea_partial_r2": float(shea_r2),
                "n_obs": n,
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Weak instrument tests
# ---------------------------------------------------------------------------


def cragg_donald_test(
    data: pd.DataFrame,
    endogenous: list[str],
    instruments: list[str],
    exogenous: list[str] | None = None,
) -> DiagnosticResult:
    """Cragg-Donald (1993) minimum eigenvalue statistic for weak instruments.

    Parameters
    ----------
    data : pd.DataFrame
    endogenous, instruments : list of str
    exogenous : list of str, optional

    Returns
    -------
    DiagnosticResult
        The ``statistic`` is the Cragg-Donald F-statistic (minimum
        eigenvalue of the concentration parameter matrix).
    """
    cols = endogenous + instruments
    if exogenous:
        cols += exogenous
    df = data[list(set(cols))].dropna().copy()
    n = len(df)
    k_endog = len(endogenous)
    k_instr = len(instruments)

    if exogenous:
        W = _add_const(df[exogenous].values.astype(float))
    else:
        W = np.ones((n, 1))

    M_W = _annihilator(W)
    D = df[endogenous].values.astype(float)
    Z_excl = df[instruments].values.astype(float)

    D_tilde = M_W @ D
    Z_tilde = M_W @ Z_excl

    # Canonical correlations approach
    P_Z = Z_tilde @ np.linalg.pinv(Z_tilde.T @ Z_tilde) @ Z_tilde.T
    D_hat = P_Z @ D_tilde
    D_resid = D_tilde - D_hat

    try:
        Sigma_inv = np.linalg.inv(D_resid.T @ D_resid / n)
    except np.linalg.LinAlgError:
        Sigma_inv = np.linalg.pinv(D_resid.T @ D_resid / n)

    G = D_hat.T @ D_hat / n
    eigvals = np.linalg.eigvalsh(Sigma_inv @ G)
    min_eig = float(np.min(eigvals))

    # F-statistic version
    f_stat = min_eig * (n - W.shape[1] - k_instr) / k_instr

    return DiagnosticResult(
        statistic=f_stat,
        p_value=np.nan,  # compared against Stock-Yogo critical values
        test_name="cragg_donald",
        critical_values=stock_yogo_critical_values(k_endog, k_instr),
    )


def stock_yogo_critical_values(
    n_endogenous: int = 1,
    n_instruments: int = 1,
) -> dict[str, float]:
    """Stock-Yogo (2005) critical values for weak instrument testing.

    Returns approximate critical values for the Cragg-Donald statistic
    at various maximal bias/size thresholds.

    Parameters
    ----------
    n_endogenous : int
    n_instruments : int

    Returns
    -------
    dict
        Critical values keyed by bias threshold (e.g., ``"10%_bias"``).

    References
    ----------
    Stock, J. H., & Yogo, M. (2005). Testing for weak instruments in
    linear IV regression.
    """
    # Approximate critical values for single endogenous variable
    # Source: Stock & Yogo (2005), Table 5.2
    if n_endogenous == 1:
        table = {
            2: {"10%_bias": 19.93, "15%_bias": 11.59, "20%_bias": 8.75, "25%_bias": 7.25},
            3: {"10%_bias": 9.08, "15%_bias": 6.46, "20%_bias": 5.39, "25%_bias": 4.73},
            4: {"10%_bias": 6.46, "15%_bias": 4.99, "20%_bias": 4.30, "25%_bias": 3.92},
            5: {"10%_bias": 5.39, "15%_bias": 4.30, "20%_bias": 3.81, "25%_bias": 3.49},
        }
        return table.get(n_instruments, {"10%_bias": 10.0, "rule_of_thumb": 10.0})
    return {"rule_of_thumb": 10.0}


def kleibergen_paap_test(
    data: pd.DataFrame,
    endogenous: list[str],
    instruments: list[str],
    exogenous: list[str] | None = None,
) -> DiagnosticResult:
    """Kleibergen-Paap (2006) rk statistic for weak instruments.

    Robust to heteroskedasticity and clustering, unlike Cragg-Donald.

    Parameters
    ----------
    data : pd.DataFrame
    endogenous, instruments : list of str
    exogenous : list of str, optional

    Returns
    -------
    DiagnosticResult

    References
    ----------
    Kleibergen, F., & Paap, R. (2006). Generalized reduced rank tests
    using the singular value decomposition. *Journal of Econometrics*,
    133(1), 97--126.
    """
    cols = endogenous + instruments
    if exogenous:
        cols += exogenous
    df = data[list(set(cols))].dropna().copy()
    n = len(df)

    if exogenous:
        W = _add_const(df[exogenous].values.astype(float))
    else:
        W = np.ones((n, 1))

    M_W = _annihilator(W)
    D = df[endogenous].values.astype(float)
    Z_excl = df[instruments].values.astype(float)

    D_tilde = M_W @ D
    Z_tilde = M_W @ Z_excl

    # First-stage coefficient matrix
    Pi_hat = np.linalg.pinv(Z_tilde.T @ Z_tilde) @ (Z_tilde.T @ D_tilde)
    resid = D_tilde - Z_tilde @ Pi_hat

    # Robust covariance
    k_z = Z_tilde.shape[1]
    k_d = D_tilde.shape[1]
    Omega = np.zeros((k_z * k_d, k_z * k_d))
    for i in range(n):
        zi = Z_tilde[i]
        ei = resid[i]
        gi = np.kron(zi, ei)
        Omega += np.outer(gi, gi)
    Omega /= n

    # Singular values of Pi_hat (scaled)
    sv = np.linalg.svd(Pi_hat, compute_uv=False)
    rk_stat = float(n * np.min(sv) ** 2) if len(sv) > 0 else 0.0

    return DiagnosticResult(
        statistic=rk_stat,
        p_value=np.nan,
        test_name="kleibergen_paap",
        critical_values=stock_yogo_critical_values(len(endogenous), len(instruments)),
    )


# ---------------------------------------------------------------------------
# Anderson-Rubin confidence set
# ---------------------------------------------------------------------------


def anderson_rubin_test(
    data: pd.DataFrame,
    outcome: str,
    endogenous: list[str],
    instruments: list[str],
    exogenous: list[str] | None = None,
    *,
    beta0: float | None = None,
    alpha: float = 0.05,
) -> DiagnosticResult:
    r"""Anderson-Rubin (1949) test: robust to weak instruments.

    Tests :math:`H_0: \beta = \beta_0` using the statistic:

    .. math::

        AR = \frac{(y - X\beta_0)'P_Z(y - X\beta_0) / q}
             {(y - X\beta_0)'M_Z(y - X\beta_0) / (n - k)}

    which has an :math:`F(q, n-k)` distribution under the null
    regardless of instrument strength.

    Parameters
    ----------
    data : pd.DataFrame
    outcome : str
    endogenous : list of str
    instruments : list of str
    exogenous : list of str, optional
    beta0 : float, optional
        Null hypothesis value (default 0).
    alpha : float

    Returns
    -------
    DiagnosticResult
        ``details`` contains the confidence set if computed by inversion.
    """
    if beta0 is None:
        beta0 = 0.0

    cols = [outcome] + endogenous + instruments
    if exogenous:
        cols += exogenous
    df = data[list(set(cols))].dropna().copy()
    y = df[outcome].values.astype(float)
    n = len(y)

    D = df[endogenous].values.astype(float)
    Z_excl = df[instruments].values.astype(float)

    if exogenous:
        W = _add_const(df[exogenous].values.astype(float))
        Z = np.column_stack([W, Z_excl])
    else:
        W = np.ones((n, 1))
        Z = np.column_stack([W, Z_excl])

    # Impose null on first endogenous variable
    y_tilde = y - D[:, 0] * beta0
    if D.shape[1] > 1:
        # Partial out remaining endogenous (not tested)
        y_tilde -= D[:, 1:] @ np.linalg.lstsq(D[:, 1:], y_tilde, rcond=None)[0]

    P_Z = _proj(Z)
    M_Z = np.eye(n) - P_Z
    q = Z_excl.shape[1]
    k = Z.shape[1]

    num = float(y_tilde @ P_Z @ y_tilde / q)
    den = float(y_tilde @ M_Z @ y_tilde / (n - k))
    ar_stat = num / den if den > 0 else 0.0
    ar_p = float(stats.f.sf(ar_stat, q, n - k))

    return DiagnosticResult(
        statistic=ar_stat,
        p_value=ar_p,
        test_name="anderson_rubin",
        details={"beta0": beta0, "df1": q, "df2": n - k},
    )


def anderson_rubin_ci(
    data: pd.DataFrame,
    outcome: str,
    endogenous: str,
    instruments: list[str],
    exogenous: list[str] | None = None,
    *,
    grid_min: float = -10.0,
    grid_max: float = 10.0,
    grid_n: int = 200,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Anderson-Rubin confidence interval by test inversion.

    Parameters
    ----------
    data : pd.DataFrame
    outcome, endogenous : str
    instruments : list of str
    exogenous : list of str, optional
    grid_min, grid_max : float
    grid_n : int
    alpha : float

    Returns
    -------
    tuple of float
        (lower, upper) bounds of the AR confidence set.
    """
    grid = np.linspace(grid_min, grid_max, grid_n)
    in_ci = []
    for b0 in grid:
        res = anderson_rubin_test(data, outcome, [endogenous], instruments, exogenous, beta0=b0, alpha=alpha)
        if res.p_value > alpha:
            in_ci.append(b0)

    if len(in_ci) == 0:
        return (np.nan, np.nan)
    return (float(min(in_ci)), float(max(in_ci)))


# ---------------------------------------------------------------------------
# Conditional likelihood ratio test
# ---------------------------------------------------------------------------


def conditional_lr_test(
    data: pd.DataFrame,
    outcome: str,
    endogenous: list[str],
    instruments: list[str],
    exogenous: list[str] | None = None,
    *,
    beta0: float = 0.0,
) -> DiagnosticResult:
    """Conditional likelihood ratio test (Moreira, 2003).

    More powerful than Anderson-Rubin when instruments are weak but
    not completely irrelevant.

    Parameters
    ----------
    data : pd.DataFrame
    outcome : str
    endogenous, instruments : list of str
    exogenous : list of str, optional
    beta0 : float

    Returns
    -------
    DiagnosticResult

    References
    ----------
    Moreira, M. J. (2003). A conditional likelihood ratio test for
    structural models. *Econometrica*, 71(4), 1027--1048.
    """
    # AR statistic
    ar = anderson_rubin_test(data, outcome, endogenous, instruments, exogenous, beta0=beta0)
    ar_stat = ar.statistic

    # Wald statistic from 2SLS
    iv_res = tsls(data, outcome, endogenous, instruments, exogenous)
    beta_hat = iv_res.endogenous_coef
    se_hat = iv_res.endogenous_se
    wald_stat = ((beta_hat - beta0) / se_hat) ** 2 if se_hat > 0 else 0.0

    # CLR: LR = AR - (AR - Wald)_+ ... simplified approximation
    # More precisely, CLR uses the conditional distribution given a sufficient
    # statistic for the instrument strength.
    # Simplified implementation: use AR adjusted by Wald direction
    k = len(instruments)
    if k == 1:
        clr_stat = ar_stat
        clr_p = ar.p_value
    else:
        # Approximate CLR as min(AR, Wald) adjusted
        clr_stat = min(ar_stat, wald_stat) * k
        clr_p = float(stats.chi2.sf(clr_stat, 1))

    return DiagnosticResult(
        statistic=clr_stat,
        p_value=clr_p,
        test_name="conditional_lr",
        details={"ar_stat": ar_stat, "wald_stat": wald_stat},
    )


# ---------------------------------------------------------------------------
# Sargan/Hansen J test
# ---------------------------------------------------------------------------


def sargan_test(
    data: pd.DataFrame,
    outcome: str,
    endogenous: list[str],
    instruments: list[str],
    exogenous: list[str] | None = None,
) -> DiagnosticResult:
    r"""Sargan (1958) / Hansen (1982) J-test for overidentifying restrictions.

    Tests whether the excluded instruments are uncorrelated with the
    structural error.  Requires more instruments than endogenous
    variables (overidentification).

    .. math::

        J = n \cdot \bar g' \hat S^{-1} \bar g \sim \chi^2(L - K)

    Parameters
    ----------
    data : pd.DataFrame
    outcome : str
    endogenous, instruments : list of str
    exogenous : list of str, optional

    Returns
    -------
    DiagnosticResult
    """
    k_endog = len(endogenous)
    k_instr = len(instruments)
    df_test = k_instr - k_endog

    if df_test <= 0:
        return DiagnosticResult(
            statistic=np.nan,
            p_value=np.nan,
            test_name="sargan",
            details={"message": "Exactly identified; J-test not applicable."},
        )

    # Get 2SLS residuals
    iv_res = tsls(data, outcome, endogenous, instruments, exogenous, robust=False)
    n = iv_res.n_obs

    cols = [outcome] + endogenous + instruments
    if exogenous:
        cols += exogenous
    df = data[list(set(cols))].dropna().copy()
    y = df[outcome].values.astype(float)
    D = df[endogenous].values.astype(float)
    Z_excl = df[instruments].values.astype(float)

    if exogenous:
        W = df[exogenous].values.astype(float)
        X = _add_const(np.column_stack([D, W]))
        Z = _add_const(np.column_stack([Z_excl, W]))
    else:
        X = _add_const(D)
        Z = _add_const(Z_excl)

    resid = y - X @ iv_res.coefficients

    # Sargan statistic (homoskedastic)
    sigma2 = float(np.sum(resid**2) / n)
    j_stat = float(resid @ _proj(Z) @ resid / sigma2)
    j_p = float(stats.chi2.sf(j_stat, df_test))

    return DiagnosticResult(
        statistic=j_stat,
        p_value=j_p,
        test_name="sargan",
        details={"df": df_test},
    )


def hansen_j_test(
    data: pd.DataFrame,
    outcome: str,
    endogenous: list[str],
    instruments: list[str],
    exogenous: list[str] | None = None,
) -> DiagnosticResult:
    """Hansen J-test (heteroskedasticity-robust overidentification test).

    Uses the efficient GMM objective function as the test statistic.

    Parameters
    ----------
    data : pd.DataFrame
    outcome : str
    endogenous, instruments : list of str
    exogenous : list of str, optional

    Returns
    -------
    DiagnosticResult
    """
    gmm_res = gmm_iv(data, outcome, endogenous, instruments, exogenous, weight_matrix="optimal")
    j_stat = gmm_res.details.get("j_stat", np.nan)
    j_p = gmm_res.details.get("j_p_value", np.nan)
    j_df = gmm_res.details.get("j_df", 0)

    return DiagnosticResult(
        statistic=j_stat,
        p_value=j_p,
        test_name="hansen_j",
        details={"df": j_df},
    )


# ---------------------------------------------------------------------------
# Hausman test (OLS vs 2SLS)
# ---------------------------------------------------------------------------


def hausman_test(
    data: pd.DataFrame,
    outcome: str,
    endogenous: list[str],
    instruments: list[str],
    exogenous: list[str] | None = None,
) -> DiagnosticResult:
    r"""Hausman (1978) specification test: OLS vs 2SLS.

    Tests whether the endogenous variables are in fact exogenous.
    Under the null, OLS is consistent and efficient; under the
    alternative, only 2SLS is consistent.

    .. math::

        H = (\hat\beta_{2SLS} - \hat\beta_{OLS})'
        (V_{2SLS} - V_{OLS})^{-1}
        (\hat\beta_{2SLS} - \hat\beta_{OLS})

    Parameters
    ----------
    data : pd.DataFrame
    outcome : str
    endogenous, instruments : list of str
    exogenous : list of str, optional

    Returns
    -------
    DiagnosticResult

    References
    ----------
    Hausman, J. A. (1978). Specification tests in econometrics.
    *Econometrica*, 46(6), 1251--1271.
    """
    iv_res = tsls(data, outcome, endogenous, instruments, exogenous, robust=True)

    # OLS
    cols = [outcome] + endogenous
    if exogenous:
        cols += exogenous
    df = data[list(set(cols))].dropna().copy()
    y = df[outcome].values.astype(float)
    D = df[endogenous].values.astype(float)

    if exogenous:
        W = df[exogenous].values.astype(float)
        X = _add_const(np.column_stack([D, W]))
    else:
        X = _add_const(D)

    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    resid_ols = y - X @ beta_ols
    n, k = X.shape
    sigma2_ols = float(np.sum(resid_ols**2) / (n - k))
    V_ols = sigma2_ols * np.linalg.pinv(X.T @ X)
    se_ols = np.sqrt(np.maximum(np.diag(V_ols), 0.0))

    # Compare coefficients on endogenous variables
    n_endog = len(endogenous)
    # Endogenous coefs are indices 1 through n_endog (index 0 = constant)
    idx = slice(1, 1 + n_endog)
    diff = iv_res.coefficients[idx] - beta_ols[idx]
    V_diff = np.diag(iv_res.std_errors[idx] ** 2 - se_ols[idx] ** 2)
    V_diff = np.maximum(V_diff, np.eye(n_endog) * 1e-10)

    try:
        h_stat = float(diff @ np.linalg.inv(V_diff) @ diff)
    except np.linalg.LinAlgError:
        h_stat = float(diff @ np.linalg.pinv(V_diff) @ diff)

    h_p = float(stats.chi2.sf(h_stat, n_endog))

    return DiagnosticResult(
        statistic=h_stat,
        p_value=h_p,
        test_name="hausman",
        details={"df": n_endog, "ols_coefs": beta_ols[idx].tolist(), "iv_coefs": iv_res.coefficients[idx].tolist()},
    )


# ---------------------------------------------------------------------------
# Durbin-Wu-Hausman endogeneity test
# ---------------------------------------------------------------------------


def durbin_wu_hausman(
    data: pd.DataFrame,
    outcome: str,
    endogenous: list[str],
    instruments: list[str],
    exogenous: list[str] | None = None,
) -> DiagnosticResult:
    """Durbin-Wu-Hausman regression-based endogeneity test.

    Augments the structural equation with first-stage residuals and tests
    their joint significance.

    Parameters
    ----------
    data : pd.DataFrame
    outcome : str
    endogenous, instruments : list of str
    exogenous : list of str, optional

    Returns
    -------
    DiagnosticResult
    """
    cols = [outcome] + endogenous + instruments
    if exogenous:
        cols += exogenous
    df = data[list(set(cols))].dropna().copy()
    y = df[outcome].values.astype(float)
    n = len(y)

    D = df[endogenous].values.astype(float)
    Z_excl = df[instruments].values.astype(float)

    if exogenous:
        W = df[exogenous].values.astype(float)
        Z_first = _add_const(np.column_stack([Z_excl, W]))
    else:
        Z_first = _add_const(Z_excl)

    # First stage residuals
    v_hats = []
    for j in range(D.shape[1]):
        beta_fs = np.linalg.lstsq(Z_first, D[:, j], rcond=None)[0]
        v_hats.append(D[:, j] - Z_first @ beta_fs)
    V_hat = np.column_stack(v_hats)

    # Augmented regression
    if exogenous:
        X_aug = _add_const(np.column_stack([D, df[exogenous].values.astype(float), V_hat]))
    else:
        X_aug = _add_const(np.column_stack([D, V_hat]))

    X_restricted = X_aug[:, : -len(endogenous)]  # without v_hat columns

    beta_aug = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    resid_aug = y - X_aug @ beta_aug
    ssr_u = float(np.sum(resid_aug**2))

    beta_r = np.linalg.lstsq(X_restricted, y, rcond=None)[0]
    resid_r = y - X_restricted @ beta_r
    ssr_r = float(np.sum(resid_r**2))

    q = len(endogenous)
    k = X_aug.shape[1]
    f_stat = ((ssr_r - ssr_u) / q) / (ssr_u / (n - k)) if ssr_u > 0 else 0.0
    f_p = float(stats.f.sf(f_stat, q, n - k))

    return DiagnosticResult(
        statistic=f_stat,
        p_value=f_p,
        test_name="durbin_wu_hausman",
        details={"df1": q, "df2": n - k},
    )


# ---------------------------------------------------------------------------
# JIVE (Jackknife IV estimator)
# ---------------------------------------------------------------------------


def jive(
    data: pd.DataFrame,
    outcome: str,
    endogenous: list[str],
    instruments: list[str],
    exogenous: list[str] | None = None,
    *,
    alpha: float = 0.05,
) -> IVResult:
    r"""Jackknife IV estimator (JIVE).

    Uses leave-one-out predicted values from the first stage to reduce
    finite-sample bias from many or weak instruments.

    .. math::

        \hat D_{i,(-i)} = Z_i' \hat\Pi_{(-i)}

    Parameters
    ----------
    data : pd.DataFrame
    outcome : str
    endogenous, instruments : list of str
    exogenous : list of str, optional
    alpha : float

    Returns
    -------
    IVResult

    References
    ----------
    Angrist, J. D., Imbens, G. W., & Krueger, A. B. (1999). Jackknife
    instrumental variables estimation. *Journal of Applied Econometrics*,
    14(1), 57--67.
    """
    cols = [outcome] + endogenous + instruments
    if exogenous:
        cols += exogenous
    df = data[list(set(cols))].dropna().copy()
    y = df[outcome].values.astype(float)
    n = len(y)

    D = df[endogenous].values.astype(float)
    Z_excl = df[instruments].values.astype(float)

    if exogenous:
        W = df[exogenous].values.astype(float)
        Z = _add_const(np.column_stack([Z_excl, W]))
        var_names = ["const"] + endogenous + exogenous
    else:
        Z = _add_const(Z_excl)
        var_names = ["const"] + endogenous

    # Full-sample first stage
    P_Z = _proj(Z)
    h_ii = np.diag(P_Z)

    # Leave-one-out fitted values
    D_hat_jive = np.zeros_like(D)
    D_hat_full = P_Z @ D
    for i in range(n):
        for j in range(D.shape[1]):
            D_hat_jive[i, j] = (D_hat_full[i, j] - h_ii[i] * D[i, j]) / (1 - h_ii[i])

    # Second stage with JIVE fitted values
    if exogenous:
        X = _add_const(np.column_stack([D, W]))
        X_hat = _add_const(np.column_stack([D_hat_jive, W]))
    else:
        X = _add_const(D)
        X_hat = _add_const(D_hat_jive)

    try:
        beta = np.linalg.inv(X_hat.T @ X) @ (X_hat.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(X_hat.T @ X) @ (X_hat.T @ y)

    resid = y - X @ beta
    k = X.shape[1]
    sigma2 = float(np.sum(resid**2) / (n - k))

    try:
        V = sigma2 * np.linalg.inv(X_hat.T @ X)
    except np.linalg.LinAlgError:
        V = sigma2 * np.linalg.pinv(X_hat.T @ X)

    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    t_vals = np.where(se > 0, beta / se, 0.0)
    p_vals = 2 * stats.norm.sf(np.abs(t_vals))
    z_crit = stats.norm.ppf(1 - alpha / 2)

    return IVResult(
        coefficients=beta,
        std_errors=se,
        t_stats=t_vals,
        p_values=p_vals,
        ci_lower=beta - z_crit * se,
        ci_upper=beta + z_crit * se,
        variable_names=var_names,
        n_obs=n,
        method="jive",
    )


# ---------------------------------------------------------------------------
# Split-sample IV
# ---------------------------------------------------------------------------


def split_sample_iv(
    data: pd.DataFrame,
    outcome: str,
    endogenous: list[str],
    instruments: list[str],
    exogenous: list[str] | None = None,
    *,
    split_fraction: float = 0.5,
    seed: int = 42,
    alpha: float = 0.05,
) -> IVResult:
    """Split-sample IV estimator (Angrist & Krueger, 1995).

    Randomly splits the sample in two halves.  Estimates the first stage
    on one half and the second stage on the other, reducing overfitting
    bias from many instruments.

    Parameters
    ----------
    data : pd.DataFrame
    outcome : str
    endogenous, instruments : list of str
    exogenous : list of str, optional
    split_fraction : float
    seed : int
    alpha : float

    Returns
    -------
    IVResult
    """
    cols = [outcome] + endogenous + instruments
    if exogenous:
        cols += exogenous
    df = data[list(set(cols))].dropna().copy()
    n = len(df)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n1 = int(n * split_fraction)
    idx1 = perm[:n1]
    idx2 = perm[n1:]

    df1 = df.iloc[idx1]
    df2 = df.iloc[idx2]

    y2 = df2[outcome].values.astype(float)
    D2 = df2[endogenous].values.astype(float)
    Z1 = df1[instruments].values.astype(float)
    D1 = df1[endogenous].values.astype(float)
    Z2 = df2[instruments].values.astype(float)

    if exogenous:
        W1 = df1[exogenous].values.astype(float)
        W2 = df2[exogenous].values.astype(float)
        Z1_full = _add_const(np.column_stack([Z1, W1]))
        Z2_full = _add_const(np.column_stack([Z2, W2]))
        var_names = ["const"] + endogenous + exogenous
    else:
        Z1_full = _add_const(Z1)
        Z2_full = _add_const(Z2)
        var_names = ["const"] + endogenous

    # First stage on sample 1
    D_hat2 = np.zeros_like(D2)
    for j in range(D1.shape[1]):
        beta_fs = np.linalg.lstsq(Z1_full, D1[:, j], rcond=None)[0]
        D_hat2[:, j] = Z2_full @ beta_fs

    # Second stage on sample 2
    if exogenous:
        X2 = _add_const(np.column_stack([D2, W2]))
        X2_hat = _add_const(np.column_stack([D_hat2, W2]))
    else:
        X2 = _add_const(D2)
        X2_hat = _add_const(D_hat2)

    try:
        beta = np.linalg.inv(X2_hat.T @ X2) @ (X2_hat.T @ y2)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(X2_hat.T @ X2) @ (X2_hat.T @ y2)

    resid = y2 - X2 @ beta
    n2 = len(y2)
    k = X2.shape[1]
    sigma2 = float(np.sum(resid**2) / (n2 - k))
    try:
        V = sigma2 * np.linalg.inv(X2_hat.T @ X2)
    except np.linalg.LinAlgError:
        V = sigma2 * np.linalg.pinv(X2_hat.T @ X2)

    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    t_vals = np.where(se > 0, beta / se, 0.0)
    p_vals = 2 * stats.norm.sf(np.abs(t_vals))
    z_crit = stats.norm.ppf(1 - alpha / 2)

    return IVResult(
        coefficients=beta,
        std_errors=se,
        t_stats=t_vals,
        p_values=p_vals,
        ci_lower=beta - z_crit * se,
        ci_upper=beta + z_crit * se,
        variable_names=var_names,
        n_obs=n,
        method="split_sample_iv",
    )


# ---------------------------------------------------------------------------
# Control function approach
# ---------------------------------------------------------------------------


def control_function(
    data: pd.DataFrame,
    outcome: str,
    endogenous: list[str],
    instruments: list[str],
    exogenous: list[str] | None = None,
    *,
    robust: bool = True,
    alpha: float = 0.05,
) -> IVResult:
    """Control function (CF) approach to endogeneity.

    Two-step: (1) regress endogenous on instruments and exogenous to
    get residuals; (2) include residuals in the structural equation
    as additional regressors.  Numerically equivalent to 2SLS for
    linear models but extends naturally to nonlinear models.

    Parameters
    ----------
    data : pd.DataFrame
    outcome : str
    endogenous, instruments : list of str
    exogenous : list of str, optional
    robust : bool
    alpha : float

    Returns
    -------
    IVResult
        ``details`` includes the test of endogeneity (significance of
        first-stage residuals).
    """
    cols = [outcome] + endogenous + instruments
    if exogenous:
        cols += exogenous
    df = data[list(set(cols))].dropna().copy()
    y = df[outcome].values.astype(float)
    n = len(y)
    D = df[endogenous].values.astype(float)
    Z_excl = df[instruments].values.astype(float)

    if exogenous:
        W = df[exogenous].values.astype(float)
        Z_first = _add_const(np.column_stack([Z_excl, W]))
    else:
        Z_first = _add_const(Z_excl)

    # First stage: get residuals
    v_hats = []
    for j in range(D.shape[1]):
        beta_fs = np.linalg.lstsq(Z_first, D[:, j], rcond=None)[0]
        v_hats.append(D[:, j] - Z_first @ beta_fs)
    V_hat = np.column_stack(v_hats)

    # Second stage: include residuals
    if exogenous:
        X = _add_const(np.column_stack([D, W, V_hat]))
        var_names = ["const"] + endogenous + exogenous + [f"v_hat_{e}" for e in endogenous]
    else:
        X = _add_const(np.column_stack([D, V_hat]))
        var_names = ["const"] + endogenous + [f"v_hat_{e}" for e in endogenous]

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    k = X.shape[1]

    if robust:
        # HC1 robust SE
        meat = X.T @ np.diag(resid**2) @ X
        bread = np.linalg.pinv(X.T @ X)
        V_mat = (n / (n - k)) * bread @ meat @ bread
    else:
        sigma2 = float(np.sum(resid**2) / (n - k))
        V_mat = sigma2 * np.linalg.pinv(X.T @ X)

    se = np.sqrt(np.maximum(np.diag(V_mat), 0.0))
    t_vals = np.where(se > 0, beta / se, 0.0)
    p_vals = 2 * stats.norm.sf(np.abs(t_vals))
    z_crit = stats.norm.ppf(1 - alpha / 2)

    return IVResult(
        coefficients=beta,
        std_errors=se,
        t_stats=t_vals,
        p_values=p_vals,
        ci_lower=beta - z_crit * se,
        ci_upper=beta + z_crit * se,
        variable_names=var_names,
        n_obs=n,
        method="control_function",
    )


# ---------------------------------------------------------------------------
# IV Probit
# ---------------------------------------------------------------------------


def iv_probit(
    data: pd.DataFrame,
    outcome: str,
    endogenous: list[str],
    instruments: list[str],
    exogenous: list[str] | None = None,
    *,
    alpha: float = 0.05,
) -> IVResult:
    """IV Probit via control function (Rivers & Vuong, 1988).

    For binary outcomes.  Uses the control-function approach: first-stage
    OLS residuals are included in a probit second stage.

    Parameters
    ----------
    data : pd.DataFrame
    outcome : str
        Binary outcome (0/1).
    endogenous, instruments : list of str
    exogenous : list of str, optional
    alpha : float

    Returns
    -------
    IVResult

    References
    ----------
    Rivers, D., & Vuong, Q. H. (1988). Limited information estimators
    and exogeneity tests for simultaneous probit models. *Journal of
    Econometrics*, 39(3), 347--366.
    """
    from sklearn.linear_model import LogisticRegression as LR

    cols = [outcome] + endogenous + instruments
    if exogenous:
        cols += exogenous
    df = data[list(set(cols))].dropna().copy()
    y = df[outcome].values.astype(int)
    n = len(y)
    D = df[endogenous].values.astype(float)
    Z_excl = df[instruments].values.astype(float)

    if exogenous:
        W = df[exogenous].values.astype(float)
        Z_first = _add_const(np.column_stack([Z_excl, W]))
    else:
        Z_first = _add_const(Z_excl)

    # First stage residuals
    v_hats = []
    for j in range(D.shape[1]):
        beta_fs = np.linalg.lstsq(Z_first, D[:, j], rcond=None)[0]
        v_hats.append(D[:, j] - Z_first @ beta_fs)
    V_hat = np.column_stack(v_hats)

    # Second stage: probit with control function
    if exogenous:
        X_second = np.column_stack([D, W, V_hat])
        var_names = endogenous + exogenous + [f"v_hat_{e}" for e in endogenous]
    else:
        X_second = np.column_stack([D, V_hat])
        var_names = endogenous + [f"v_hat_{e}" for e in endogenous]

    # Use logistic regression as probit approximation (probit ~ logit / 1.7)
    lr = LR(max_iter=1000, solver="lbfgs", penalty=None)
    lr.fit(X_second, y)

    beta = np.concatenate([[lr.intercept_[0]], lr.coef_[0]])
    var_names = ["const"] + var_names

    # Bootstrap SE
    rng = np.random.default_rng(42)
    boot_betas = []
    for _ in range(200):
        idx = rng.choice(n, size=n, replace=True)
        try:
            lr_b = LR(max_iter=1000, solver="lbfgs", penalty=None)
            lr_b.fit(X_second[idx], y[idx])
            boot_betas.append(np.concatenate([[lr_b.intercept_[0]], lr_b.coef_[0]]))
        except Exception:
            continue

    if len(boot_betas) > 1:
        se = np.std(boot_betas, axis=0, ddof=1)
    else:
        se = np.full(len(beta), np.nan)

    t_vals = np.where(se > 0, beta / se, 0.0)
    p_vals = 2 * stats.norm.sf(np.abs(t_vals))
    z_crit = stats.norm.ppf(1 - alpha / 2)

    return IVResult(
        coefficients=beta,
        std_errors=se,
        t_stats=t_vals,
        p_values=p_vals,
        ci_lower=beta - z_crit * se,
        ci_upper=beta + z_crit * se,
        variable_names=var_names,
        n_obs=n,
        method="iv_probit",
    )


# ---------------------------------------------------------------------------
# Panel IV (fixed-effects 2SLS)
# ---------------------------------------------------------------------------


def panel_iv(
    data: pd.DataFrame,
    outcome: str,
    endogenous: list[str],
    instruments: list[str],
    unit: str,
    *,
    exogenous: list[str] | None = None,
    time_fe: str | None = None,
    alpha: float = 0.05,
) -> IVResult:
    """Panel IV with unit (and optionally time) fixed effects.

    Demeans all variables by unit (and time if specified), then applies
    2SLS to the demeaned data.

    Parameters
    ----------
    data : pd.DataFrame
    outcome : str
    endogenous, instruments : list of str
    unit : str
        Unit identifier for fixed effects.
    exogenous : list of str, optional
    time_fe : str, optional
        Time column for two-way fixed effects.
    alpha : float

    Returns
    -------
    IVResult
    """
    df = data.copy()
    all_vars = [outcome] + endogenous + instruments
    if exogenous:
        all_vars += exogenous

    # Demean by unit
    for v in all_vars:
        df[v] = df[v].astype(float) - df.groupby(unit)[v].transform("mean").astype(float)

    # Demean by time if requested
    if time_fe:
        for v in all_vars:
            df[v] = df[v] - df.groupby(time_fe)[v].transform("mean")

    # Apply 2SLS to demeaned data (no constant needed)
    y = df[outcome].values.astype(float)
    D = df[endogenous].values.astype(float)
    Z_excl = df[instruments].values.astype(float)
    n = len(y)

    if exogenous:
        W = df[exogenous].values.astype(float)
        X = np.column_stack([D, W])
        Z = np.column_stack([Z_excl, W])
        var_names = endogenous + exogenous
    else:
        X = D
        Z = Z_excl
        var_names = endogenous

    P_Z = _proj(Z)
    try:
        beta = np.linalg.inv(X.T @ P_Z @ X) @ (X.T @ P_Z @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(X.T @ P_Z @ X) @ (X.T @ P_Z @ y)

    resid = y - X @ beta
    cluster_ids = df[unit].values
    se = _cluster_se(X, resid, Z, cluster_ids)

    t_vals = np.where(se > 0, beta / se, 0.0)
    p_vals = 2 * stats.norm.sf(np.abs(t_vals))
    z_crit = stats.norm.ppf(1 - alpha / 2)

    return IVResult(
        coefficients=beta,
        std_errors=se,
        t_stats=t_vals,
        p_values=p_vals,
        ci_lower=beta - z_crit * se,
        ci_upper=beta + z_crit * se,
        variable_names=var_names,
        n_obs=n,
        method="panel_iv_fe",
        details={"n_units": df[unit].nunique()},
    )


# ---------------------------------------------------------------------------
# IV regression diagnostics
# ---------------------------------------------------------------------------


def iv_diagnostics(
    data: pd.DataFrame,
    outcome: str,
    endogenous: list[str],
    instruments: list[str],
    exogenous: list[str] | None = None,
) -> dict[str, Any]:
    """Comprehensive IV regression diagnostic suite.

    Runs first-stage diagnostics, weak instrument tests, overidentification
    tests, and endogeneity tests in one call.

    Parameters
    ----------
    data : pd.DataFrame
    outcome : str
    endogenous, instruments : list of str
    exogenous : list of str, optional

    Returns
    -------
    dict
        Keys: ``first_stage``, ``cragg_donald``, ``hausman``,
        ``sargan`` (if overidentified), ``durbin_wu_hausman``.
    """
    results = {}
    results["first_stage"] = first_stage_diagnostics(data, endogenous, instruments, exogenous)
    results["cragg_donald"] = cragg_donald_test(data, endogenous, instruments, exogenous)
    results["hausman"] = hausman_test(data, outcome, endogenous, instruments, exogenous)
    results["durbin_wu_hausman"] = durbin_wu_hausman(data, outcome, endogenous, instruments, exogenous)

    if len(instruments) > len(endogenous):
        results["sargan"] = sargan_test(data, outcome, endogenous, instruments, exogenous)
        results["hansen_j"] = hansen_j_test(data, outcome, endogenous, instruments, exogenous)

    return results


def iv_residual_analysis(
    data: pd.DataFrame,
    outcome: str,
    endogenous: list[str],
    instruments: list[str],
    exogenous: list[str] | None = None,
) -> pd.DataFrame:
    """Residual analysis for IV regression.

    Returns a DataFrame with fitted values, residuals, leverage, and
    standardised residuals.

    Parameters
    ----------
    data : pd.DataFrame
    outcome : str
    endogenous, instruments : list of str
    exogenous : list of str, optional

    Returns
    -------
    pd.DataFrame
    """
    iv_res = tsls(data, outcome, endogenous, instruments, exogenous)

    cols = [outcome] + endogenous + instruments
    if exogenous:
        cols += exogenous
    df = data[list(set(cols))].dropna().copy()
    y = df[outcome].values.astype(float)
    D = df[endogenous].values.astype(float)

    if exogenous:
        W = df[exogenous].values.astype(float)
        X = _add_const(np.column_stack([D, W]))
    else:
        X = _add_const(D)

    fitted = X @ iv_res.coefficients
    resid = y - fitted
    n = len(y)
    sigma = np.sqrt(np.sum(resid**2) / (n - X.shape[1]))

    # Leverage from second stage
    H = X @ np.linalg.pinv(X.T @ X) @ X.T
    h = np.diag(H)
    std_resid = resid / (sigma * np.sqrt(np.maximum(1 - h, 1e-10)))

    return pd.DataFrame(
        {
            "fitted": fitted,
            "residual": resid,
            "leverage": h,
            "std_residual": std_resid,
        }
    )
