"""morie.otis_causal -- IPW, AIPW, and DML estimators for OTIS.

Companion to ``morie.otis.otdml`` (which already implements
cross-fitted Frisch-Waugh-Lovell DML).  This module adds:

  * :func:`otis_ipw`   -- Hájek-stabilised inverse probability weighting
    with a logistic propensity model.
  * :func:`otis_aipw`  -- Augmented IPW (Robins-Rotnitzky-Zhao 1994
    doubly-robust estimator) with cross-fitted nuisance models.
  * :func:`otis_causal_grid` -- runs all three estimators (IPW, AIPW,
    DML) on the canonical three (treatment, outcome) pairs:
        (a) MentalHealth_Alert -> SuicideRisk_Alert
        (b) HighAlertComplexity -> AnyReadmission
        (c) RegionalVolatility -> ConsecutiveSegregationDays

The three pairs cover the three Goffmanian framings of OTIS most
relevant to the MA-paired thesis: clinical-alert chain (a),
multi-alert "complexity" (b), and inter-region churn (c).

Estimators
----------
Let $D \\in \\{0,1\\}$ be a binary treatment, $Y$ a real-valued
outcome, $X \\in \\mathbb{R}^p$ a covariate vector, and write
$e(X) \\coloneqq \\Pr(D=1\\mid X)$ for the propensity score and
$\\mu_d(X) \\coloneqq \\mathbb{E}[Y\\mid D=d, X]$ for the
outcome-regression mean.

IPW (Hájek):

    ATE_IPW = sum_i w1(i)·D_i·Y_i / sum_i w1(i)·D_i
              - sum_i w0(i)·(1-D_i)·Y_i / sum_i w0(i)·(1-D_i)

with w1(i) = 1/e(X_i), w0(i) = 1/(1-e(X_i)).

AIPW (RRZ 1994 doubly-robust score):

    psi(O_i; eta) = mu_1(X_i) - mu_0(X_i)
                    + D_i·(Y_i - mu_1(X_i))/e(X_i)
                    - (1-D_i)·(Y_i - mu_0(X_i))/(1-e(X_i))

    ATE_AIPW = (1/n) sum_i psi(O_i; eta_hat)

with cross-fitted nuisance estimates eta_hat = (e_hat, mu_1_hat,
mu_0_hat).

DML PLR:

    Frisch-Waugh-Lovell partialled-out regression with cross-fitted
    residuals (already implemented in :func:`morie.otis.otdml`).

References
----------
Robins, J. M., Rotnitzky, A., and Zhao, L. P. (1994).  Estimation of
regression coefficients when some regressors are not always observed.
J. Amer. Statist. Assoc. 89(427): 846-866.

Chernozhukov, V. et al. (2018).  Double/debiased machine learning for
treatment and structural parameters.  Econometrics Journal 21(1):
C1-C68.

Lunceford, J. K. and Davidian, M. (2004).  Stratification and
weighting via the propensity score in estimation of causal treatment
effects: a comparative study.  Statistics in Medicine 23(19):
2937-2960.

Robins, J. M. (1986).  A new approach to causal inference in
mortality studies with a sustained exposure period -- application
to control of the healthy worker survivor effect.  Mathematical
Modelling 7: 1393-1512.

Rosenbaum, P. R. and Rubin, D. B. (1983).  The central role of the
propensity score in observational studies for causal effects.
Biometrika 70(1): 41-55.

Attribution -- Ruhela formulations
---------------------------------
The OTIS-RC formulations and the entire codebase implemented here
are the work of Vansh Singh Ruhela (hadesllm).  No one
co-formulated the analyses: the (T = ac ≥ 2, Y = vm count)
alert-complexity -> regional-volatility contrast, the 8-state
combo encoding, and the full method battery (IPW + AIPW +
g-computation + PSM-NN + PSM-subclass + IRM-DML + match_first)
are Ruhela's design.

Three external contributors enabled the journey but did not
co-author the formulations:

  Doob       -- Prof. Emeritus Anthony N. Doob, University of Toronto.
                Pointed Ruhela to the existence of the OTIS data feed --
                the first piece of the puzzle.  Without that pointer,
                this project does not begin.
  Levinsky   -- Prof. Zachary Levinsky, University of Toronto.  His
                graduate course on the Theory of Punishment was the
                broader context that motivated this work.  He later
                reviewed the preliminary a01-only analysis from the
                ``OTIS-RC/`` folder; he did not see the full method
                battery shipped here.
  Medina     -- Prof. Zorro Medina, University of Toronto.  Reviewed
                the formal write-up; her feedback validated the
                statistical route and gave the pipeline confidence in
                its current form.

Naming conventions
------------------
  RF  -- Ruhela formulation: a (treatment, outcome, covariates) design
         choice for a specific OTIS dataset.
  RDF -- Ruhela Dual Formulation: an RF paired with a Naive-arm
         sensitivity contrast (e.g., on a01/b01 the Ruhela arm
         T_high_ac -> vm_count is paired with the Naive arm
         any-flag -> vm-binary).
  DLRM -- Doob-Levinsky-Ruhela-Medina, short alias for the
         attribution chain on the methodology side. Function names
         like ``analyze_a01_dlrm`` are equivalent aliases for
         ``analyze_a01_ruhela_formulations``.

ACKNOWLEDGEMENTS -- beyond the methodology attribution above
-----------------------------------------------------------
The DLRM attribution covers the contributors whose work directly
shaped the OTIS-RC analyses. The wider intellectual journey that
brought this codebase into being also owes credit to:

  Prof. Beatrice Jauregui, University of Toronto -- for the network
    of mentorship that connected Ruhela to Prof. Medina, which made
    this line of work possible.

  Prof. Ayobami Laniyonu, University of Toronto -- for valuable
    lessons over the broader period of this work that informed
    Ruhela's thinking on policing and corrections research.

The separation between methodology attribution (DLRM, narrow) and
acknowledgements (broader, enabling) follows standard academic-
publishing convention.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, asdict
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats as sps

from .fn._richresult import RichResult


# ── Estimator return type ──────────────────────────────────────────


@dataclass
class CausalEstimate:
    estimator: str  # "IPW" | "AIPW" | "DML" | "g-computation" | "PSM" | "PSM-subclass"
    ate: float
    ate_se: float
    ate_pval: float
    n: int
    n_treated: int
    p_treat: float
    notes: list[str] = field(default_factory=list)

    @property
    def ate_ci95(self) -> tuple[float, float]:
        return (self.ate - 1.96 * self.ate_se,
                self.ate + 1.96 * self.ate_se)


# ── Utilities ──────────────────────────────────────────────────────


def _binarise(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_string_dtype(s):
        return (s.astype(str).str.strip().str.lower() == "yes").astype(int).to_numpy()
    return s.fillna(0).astype(int).to_numpy()


def _design_matrix(data: pd.DataFrame, covariates: list[str]) -> np.ndarray:
    """Build a numeric design matrix (intercept + dummies) from ``covariates``."""
    X = data[covariates].copy()
    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    X = X.astype(np.float64)
    X.insert(0, "_intercept", 1.0)
    return X.to_numpy()


def _logit_fit(X: np.ndarray, d: np.ndarray, ridge: float = 1e-3,
                max_iter: int = 50, tol: float = 1e-6) -> np.ndarray:
    """Newton-Raphson MLE for logistic regression with light ridge.

    Avoids a sklearn dependency.  Ridge stabilises perfect-separation
    cases (rare cells in the OTIS interaction structure).
    """
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        eta = X @ beta
        eta = np.clip(eta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = mu * (1 - mu) + 1e-9
        H = X.T @ (w[:, None] * X) + ridge * np.eye(p)
        g = X.T @ (d - mu) - ridge * beta
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        beta_new = beta + step
        if np.max(np.abs(step)) < tol:
            return beta_new
        beta = beta_new
    return beta


def _propensity_clip(e: np.ndarray, eps: float = 0.02) -> np.ndarray:
    """Clip propensities away from 0/1 for IPW / AIPW stability."""
    return np.clip(e, eps, 1 - eps)


# ── IPW (Hájek) ─────────────────────────────────────────────────────


def otis_ipw(df: pd.DataFrame, *,
              treatment: str, outcome: str,
              covariates: list[str],
              eps: float = 0.02,
              propensity_calibration: str = "none",
              ) -> CausalEstimate:
    """Hájek-stabilised IPW estimator of the ATE on OTIS data.

    Estimates the propensity by logistic regression and weights
    treated/control outcomes by $1/\\hat e$ and $1/(1-\\hat e)$.
    Standard error follows the Lunceford-Davidian sandwich.

    Parameters
    ----------
    propensity_calibration : "none" | "platt" | "isotonic"
        Calibrate the raw logistic propensities. Platt = sigmoid(a + b * logit(p))
        fitted on the same data; isotonic = monotonic regression
        (sklearn). Calibration improves Brier score / calibration-curve
        diagnostics; for OTIS data with logistic propensities it usually
        moves the predicted prevalence toward the observed prevalence.
    """
    data = df[[treatment, outcome] + covariates].dropna().copy()
    d = _binarise(data[treatment])
    y = data[outcome].astype(np.float64).to_numpy()
    X = _design_matrix(data, covariates)
    n = y.size
    n_treated = int(d.sum())
    p_treat = float(d.mean())

    beta = _logit_fit(X, d)
    eta = np.clip(X @ beta, -30, 30)
    e = _propensity_clip(1.0 / (1.0 + np.exp(-eta)), eps=eps)

    if propensity_calibration != "none":
        e = _calibrate_propensity(e, d, method=propensity_calibration)
        e = _propensity_clip(e, eps=eps)

    # Hájek normalisation
    w1 = d / e
    w0 = (1 - d) / (1 - e)
    mu1 = (w1 * y).sum() / w1.sum()
    mu0 = (w0 * y).sum() / w0.sum()
    ate = float(mu1 - mu0)

    # Sandwich SE (Lunceford & Davidian 2004 §3)
    influence = (d * (y - mu1) / e) - ((1 - d) * (y - mu0) / (1 - e))
    se = float(influence.std(ddof=1) / math.sqrt(n))
    z = ate / se if se > 0 else 0.0
    pval = float(2 * (1 - sps.norm.cdf(abs(z))))

    notes = [f"calibration={propensity_calibration}"]
    n_clipped = int(((e <= eps + 1e-9) | (e >= 1 - eps - 1e-9)).sum())
    if n_clipped > 0:
        notes.append(f"{n_clipped} propensities clipped")
    diag = _propensity_diagnostics(e, d)
    notes.append(f"Brier={diag['brier']:.3f}")
    return CausalEstimate("IPW", ate, se, pval, n, n_treated, p_treat, notes)


# ── AIPW (cross-fitted, RRZ 1994 / DR ATE) ─────────────────────────


def otis_aipw(df: pd.DataFrame, *,
               treatment: str, outcome: str,
               covariates: list[str],
               n_folds: int = 5, seed: int = 123,
               eps: float = 0.02,
               propensity_calibration: str = "none",
               ) -> CausalEstimate:
    """Augmented IPW (Robins-Rotnitzky-Zhao) ATE on OTIS data.

    Uses ``n_folds`` cross-fitting: nuisance models (propensity,
    outcome-regression) are fit on $K-1$ folds and evaluated on the
    held-out fold, then aggregated.  This is the doubly-robust
    influence-function plug-in estimator.

    Parameters
    ----------
    propensity_calibration : "none" | "platt" | "isotonic"
        Calibrate the cross-fitted propensity scores within each
        fold's test set against its training labels. Improves Brier-
        score / calibration-curve diagnostics.
    """
    data = df[[treatment, outcome] + covariates].dropna().copy()
    d = _binarise(data[treatment])
    y = data[outcome].astype(np.float64).to_numpy()
    X = _design_matrix(data, covariates)
    n = y.size
    n_treated = int(d.sum())
    p_treat = float(d.mean())

    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), n_folds)
    e_hat = np.empty(n)
    mu1_hat = np.empty(n)
    mu0_hat = np.empty(n)

    for k in range(n_folds):
        test = folds[k]
        train = np.setdiff1d(np.arange(n), test)
        # Propensity (logistic)
        beta_p = _logit_fit(X[train], d[train])
        eta = np.clip(X[test] @ beta_p, -30, 30)
        e_hat[test] = _propensity_clip(1.0 / (1.0 + np.exp(-eta)), eps=eps)
        # Outcome regression: separately for D=1 and D=0
        for dval, mu in ((1, mu1_hat), (0, mu0_hat)):
            mask = train[d[train] == dval]
            if mask.size < X.shape[1] + 2:
                # Not enough data for OLS; use mean
                mu[test] = y[mask].mean() if mask.size else y[train].mean()
                continue
            beta_y, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
            mu[test] = X[test] @ beta_y

    if propensity_calibration != "none":
        e_hat = _calibrate_propensity(e_hat, d, method=propensity_calibration)
        e_hat = _propensity_clip(e_hat, eps=eps)

    # Doubly-robust influence function
    psi = (mu1_hat - mu0_hat
           + d * (y - mu1_hat) / e_hat
           - (1 - d) * (y - mu0_hat) / (1 - e_hat))
    ate = float(psi.mean())
    se = float(psi.std(ddof=1) / math.sqrt(n))
    z = ate / se if se > 0 else 0.0
    pval = float(2 * (1 - sps.norm.cdf(abs(z))))

    notes = [f"cross-fit folds={n_folds}",
             f"calibration={propensity_calibration}"]
    n_clipped = int(((e_hat <= eps + 1e-9)
                       | (e_hat >= 1 - eps - 1e-9)).sum())
    if n_clipped > 0:
        notes.append(f"{n_clipped} propensities clipped")
    diag = _propensity_diagnostics(e_hat, d)
    notes.append(f"Brier={diag['brier']:.3f}")
    return CausalEstimate("AIPW", ate, se, pval, n, n_treated, p_treat, notes)



def otis_gcomputation(df: pd.DataFrame, *,
                      treatment: str, outcome: str,
                      covariates: list[str],
                      n_bootstrap: int = 200,
                      seed: int = 123) -> CausalEstimate:
    """Parametric g-computation (Robins 1986) of the ATE on OTIS data.

    Fits an outcome regression $\\\\hat\\\\mu(d, X)$ via OLS, then computes
    the standardised mean difference

        \\\\hat\\\\Delta = n^{-1} \\\\sum_i (\\\\hat\\\\mu(1, X_i)
                                       - \\\\hat\\\\mu(0, X_i)).

    SE via non-parametric bootstrap (resample rows, refit, recompute).
    Complements IPW and AIPW: g-computation is consistent if and only
    if the outcome model is correctly specified (single-robust); IPW
    is consistent iff the propensity is correct; AIPW is doubly robust.

    References
    ----------
    Robins, J. M. (1986). A new approach to causal inference in
      mortality studies with a sustained exposure period -- application
      to control of the healthy worker survivor effect. Mathematical
      Modelling 7: 1393-1512.
    """
    data = df[[treatment, outcome] + covariates].dropna().copy()
    d = _binarise(data[treatment])
    y = data[outcome].astype(np.float64).to_numpy()
    X = _design_matrix(data, covariates)
    n = y.size
    n_treated = int(d.sum())
    p_treat = float(d.mean())

    # Outcome regression with treatment as a column
    Xfull = np.hstack([d.reshape(-1, 1), X])
    beta, *_ = np.linalg.lstsq(Xfull, y, rcond=None)
    # Counterfactual predictions
    X1 = np.hstack([np.ones((n, 1)), X])
    X0 = np.hstack([np.zeros((n, 1)), X])
    mu1 = X1 @ beta
    mu0 = X0 @ beta
    ate = float((mu1 - mu0).mean())

    # Non-parametric bootstrap SE
    rng = np.random.default_rng(seed)
    boot_ates = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        try:
            beta_b, *_ = np.linalg.lstsq(Xfull[idx], y[idx], rcond=None)
            mu1_b = X1 @ beta_b
            mu0_b = X0 @ beta_b
            boot_ates[b] = (mu1_b - mu0_b).mean()
        except Exception:  # noqa: BLE001
            boot_ates[b] = np.nan
    boot_ates = boot_ates[~np.isnan(boot_ates)]
    se = float(boot_ates.std(ddof=1)) if boot_ates.size > 1 else float("nan")
    z = ate / se if se > 0 else 0.0
    pval = float(2 * (1 - sps.norm.cdf(abs(z)))) if se > 0 else float("nan")

    notes = [f"bootstrap={n_bootstrap}",
             f"valid_bootstrap_replicates={boot_ates.size}"]
    return CausalEstimate("g-computation", ate, se, pval,
                            n, n_treated, p_treat, notes)


def otis_psm_subclass(df: pd.DataFrame, *,
                      treatment: str, outcome: str,
                      covariates: list[str],
                      n_strata: int = 5,
                      eps: float = 0.02) -> CausalEstimate:
    """Propensity-score stratification (Rosenbaum & Rubin 1983).

    Splits the sample into ``n_strata`` strata by propensity-score
    quantile, computes within-stratum ATE = E[Y|D=1, S=s] - E[Y|D=0, S=s],
    and weights stratum-specific ATEs by stratum size to get the
    population ATE. Standard error: weighted average of within-stratum
    SEs (Rosenbaum-Rubin convention).

    n_strata=5 is the standard "rule of thumb" -- Cochran (1968) showed
    5 strata remove ~90% of bias from a single confounder.

    References
    ----------
    Rosenbaum, P. R. & Rubin, D. B. (1983). The central role of the
      propensity score in observational studies for causal effects.
      Biometrika 70(1): 41-55.
    Cochran, W. G. (1968). The effectiveness of adjustment by
      subclassification in removing bias in observational studies.
      Biometrics 24(2): 295-313.
    """
    data = df[[treatment, outcome] + covariates].dropna().copy()
    d = _binarise(data[treatment])
    y = data[outcome].astype(np.float64).to_numpy()
    X = _design_matrix(data, covariates)
    n = y.size
    n_treated = int(d.sum())
    p_treat = float(d.mean())

    beta = _logit_fit(X, d)
    eta = np.clip(X @ beta, -30, 30)
    e = _propensity_clip(1.0 / (1.0 + np.exp(-eta)), eps=eps)

    # Quantile-based strata
    qs = np.quantile(e, np.linspace(0, 1, n_strata + 1))
    qs[0] = -np.inf  # ensure first stratum captures left tail
    qs[-1] = np.inf
    stratum = np.searchsorted(qs[1:-1], e, side="right")  # 0..n_strata-1

    # Within-stratum ATE + variance
    deltas = np.full(n_strata, np.nan)
    vars_ = np.full(n_strata, np.nan)
    weights = np.zeros(n_strata)
    n_dropped = 0
    for s in range(n_strata):
        mask = stratum == s
        ds = d[mask]
        ys = y[mask]
        if mask.sum() == 0 or ds.sum() == 0 or (1 - ds).sum() == 0:
            n_dropped += int(mask.sum())
            continue
        y1 = ys[ds == 1]
        y0 = ys[ds == 0]
        deltas[s] = float(y1.mean() - y0.mean())
        var1 = float(y1.var(ddof=1) / max(y1.size, 1)) if y1.size > 1 else 0.0
        var0 = float(y0.var(ddof=1) / max(y0.size, 1)) if y0.size > 1 else 0.0
        vars_[s] = var1 + var0
        weights[s] = mask.sum()

    valid = ~np.isnan(deltas)
    if not valid.any():
        return CausalEstimate("PSM-subclass", float("nan"), float("nan"),
                                1.0, n, n_treated, p_treat,
                                notes=[f"no valid strata (n_strata={n_strata})"])
    w = weights[valid] / weights[valid].sum()
    ate = float((w * deltas[valid]).sum())
    se = float(np.sqrt(((w ** 2) * vars_[valid]).sum()))
    z = ate / se if se > 0 else 0.0
    pval = float(2 * (1 - sps.norm.cdf(abs(z))))

    notes = [f"strata={n_strata}", f"valid_strata={int(valid.sum())}"]
    if n_dropped > 0:
        notes.append(f"{n_dropped} units in single-arm strata dropped")
    return CausalEstimate("PSM-subclass", ate, se, pval,
                            n, n_treated, p_treat, notes)



def otis_atc(df: pd.DataFrame, *,
             treatment: str, outcome: str,
             covariates: list[str],
             n_folds: int = 5, seed: int = 123,
             eps: float = 0.02) -> CausalEstimate:
    """AIPW-flavoured Average Treatment effect on the Controls (ATC).

    The IF for ATC = E[Y(1) - Y(0) | D = 0] is

      psi^{ATC}(O; eta) = mu_1(X) - mu_0(X)
                          + (1-D)/(1-e(X)) * (Y - mu_0(X))
                          - e(X)/(1-e(X)) * D * (Y - mu_1(X)) / e(X)

    weighted by 1 / Pr(D=0). The implementation reuses the cross-fitted
    nuisance pipeline of `otis_aipw` and reweights the score on the
    untreated stratum.
    """
    data = df[[treatment, outcome] + covariates].dropna().copy()
    d = _binarise(data[treatment])
    y = data[outcome].astype(np.float64).to_numpy()
    X = _design_matrix(data, covariates)
    n = y.size
    n_treated = int(d.sum())
    p_treat = float(d.mean())
    p_d0 = max(1.0 - p_treat, 1e-9)

    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), n_folds)
    e_hat = np.empty(n)
    mu1_hat = np.empty(n)
    mu0_hat = np.empty(n)
    for k in range(n_folds):
        test = folds[k]
        train = np.setdiff1d(np.arange(n), test)
        beta_p = _logit_fit(X[train], d[train])
        eta = np.clip(X[test] @ beta_p, -30, 30)
        e_hat[test] = _propensity_clip(1.0 / (1.0 + np.exp(-eta)), eps=eps)
        for dval, mu in ((1, mu1_hat), (0, mu0_hat)):
            mask = train[d[train] == dval]
            if mask.size < X.shape[1] + 2:
                mu[test] = y[mask].mean() if mask.size else y[train].mean()
                continue
            beta_y, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
            mu[test] = X[test] @ beta_y

    # ATC influence function on the D=0 weighted score
    # E[Y(1)-Y(0) | D=0] estimated by:
    #   atc = (1/n_d0) * sum_i (1-D_i) * (mu_1(X_i) - Y_i)
    #         + sum_i D_i * (1-e(X_i))/e(X_i) * (Y_i - mu_1(X_i)) / n_d0
    # We use the Robins-style efficient form below.
    psi = ((1 - d) * (mu1_hat - mu0_hat)
            + d * ((1 - e_hat) / e_hat) * (y - mu1_hat)
            - (1 - d) * (y - mu0_hat)) / p_d0
    atc = float(psi.mean())
    se = float(psi.std(ddof=1) / math.sqrt(n))
    z = atc / se if se > 0 else 0.0
    pval = float(2 * (1 - sps.norm.cdf(abs(z))))
    notes = [f"cross-fit folds={n_folds}",
             f"control n={int((1-d).sum())}",
             f"E[Y(1)-Y(0) | D=0]"]
    return CausalEstimate("ATC", atc, se, pval, n, n_treated, p_treat, notes)


def otis_plr(df: pd.DataFrame, *,
             treatment: str, outcome: str,
             covariates: list[str],
             n_folds: int = 3, seed: int = 123,
             ml_outcome: str = "rf",
             ml_treatment: str = "rf") -> CausalEstimate:
    """Partially Linear Regression DML (Chernozhukov et al. 2018).

    Fits the structural model

        Y = θ * D + g(X) + ε,    D = m(X) + V

    via Frisch-Waugh-Lovell partialling-out: cross-fit nuisances
    g(X) = E[Y|X] and m(X) = E[D|X], then regress the residualised
    Y on the residualised D to recover θ.

    Cross-fitted random-forest nuisance models match the OTIS-RC
    DoubleML/PLR pipeline. Differs from IRM-DML in that PLR assumes a
    constant (homogeneous) treatment effect θ; IRM allows D × X
    interactions in the outcome model. PLR is the simpler / more
    restricted estimator. Concordance with IRM-DML is the standard
    "no effect heterogeneity" robustness signal.
    """
    data = df[[treatment, outcome] + covariates].dropna().copy()
    d = _binarise(data[treatment]).astype(np.float64)
    y = data[outcome].astype(np.float64).to_numpy()
    X = _design_matrix(data, covariates)
    n = y.size
    n_treated = int(d.sum())
    p_treat = float(d.mean())

    HAS_RF = False
    if ml_outcome == "rf" or ml_treatment == "rf":
        try:
            from sklearn.ensemble import (RandomForestRegressor,
                                            RandomForestClassifier)
            HAS_RF = True
        except ImportError:
            HAS_RF = False

    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), n_folds)
    g_hat = np.empty(n)  # E[Y|X]
    m_hat = np.empty(n)  # E[D|X]

    for k in range(n_folds):
        test = folds[k]
        train = np.setdiff1d(np.arange(n), test)
        if HAS_RF and ml_outcome == "rf":
            reg_g = RandomForestRegressor(
                n_estimators=500, max_depth=5,
                random_state=seed + k, n_jobs=-1)
            reg_g.fit(X[train], y[train])
            g_hat[test] = reg_g.predict(X[test])
        else:
            beta_g, *_ = np.linalg.lstsq(X[train], y[train], rcond=None)
            g_hat[test] = X[test] @ beta_g
        if HAS_RF and ml_treatment == "rf":
            reg_m = RandomForestClassifier(
                n_estimators=500, max_depth=5,
                random_state=seed + k * 7, n_jobs=-1)
            reg_m.fit(X[train], d[train].astype(int))
            m_hat[test] = reg_m.predict_proba(X[test])[:, 1]
        else:
            beta_m = _logit_fit(X[train], d[train])
            eta_m = np.clip(X[test] @ beta_m, -30, 30)
            m_hat[test] = 1.0 / (1.0 + np.exp(-eta_m))

    # Residualised regression
    y_resid = y - g_hat
    d_resid = d - m_hat
    var_d = float((d_resid ** 2).mean())
    if var_d < 1e-12:
        return CausalEstimate("PLR", float("nan"), float("nan"), 1.0,
                                n, n_treated, p_treat,
                                notes=["zero residual variance in D"])
    theta = float((d_resid * y_resid).mean() / var_d)
    # Sandwich SE on the score psi_i = (Y_i - g - theta * (D_i - m)) * (D_i - m)
    psi = (y_resid - theta * d_resid) * d_resid
    sigma2 = float((psi ** 2).mean())
    se = float(math.sqrt(sigma2 / (n * var_d ** 2)))
    z = theta / se if se > 0 else 0.0
    pval = float(2 * (1 - sps.norm.cdf(abs(z))))
    notes = [f"cross-fit folds={n_folds}",
             f"residual var(D|X)={var_d:.4f}",
             "homogeneous-effect assumption (θ constant)",
             f"ml_outcome={ml_outcome if HAS_RF else 'ols'}",
             f"ml_treatment={ml_treatment if HAS_RF else 'logit'}"]
    return CausalEstimate("PLR", theta, se, pval, n, n_treated, p_treat,
                            notes)


def _calibrate_propensity(p_raw: np.ndarray, d: np.ndarray, *,
                            method: str = "platt") -> np.ndarray:
    """Calibrate raw propensities to better match observed D rates.

    Parameters
    ----------
    p_raw : raw propensities in (0, 1)
    d : binary treatment vector
    method : 'platt' (logistic regression on logit p_raw -> d) or
              'isotonic' (sklearn isotonic regression).

    Returns calibrated propensities. Falls back to raw if the
    calibration fit is degenerate.
    """
    if method == "none":
        return p_raw
    if method == "platt":
        # Platt: fit logistic on logit(p_raw) -> d, then transform.
        eps = 1e-6
        p_clip = np.clip(p_raw, eps, 1 - eps)
        z = np.log(p_clip / (1 - p_clip)).reshape(-1, 1)
        try:
            X1 = np.hstack([np.ones((len(z), 1)), z])
            beta = _logit_fit(X1, d.astype(int))
            eta = np.clip(X1 @ beta, -30, 30)
            return 1.0 / (1.0 + np.exp(-eta))
        except Exception:  # noqa: BLE001
            return p_raw
    if method == "isotonic":
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError:
            return p_raw
        try:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            return iso.fit_transform(p_raw, d.astype(int))
        except Exception:  # noqa: BLE001
            return p_raw
    raise ValueError(f"unknown calibration method: {method!r}")


def _propensity_diagnostics(p: np.ndarray, d: np.ndarray) -> dict:
    """Brier score + mean prevalence vs predicted vs observed."""
    brier = float(((p - d) ** 2).mean())
    obs = float(d.mean())
    pred = float(p.mean())
    # Hosmer-Lemeshow-style decile log-loss approximation
    eps = 1e-12
    p_c = np.clip(p, eps, 1 - eps)
    log_loss = float(-(d * np.log(p_c) + (1 - d) * np.log(1 - p_c)).mean())
    return {"brier": brier, "obs_prevalence": obs,
            "predicted_prevalence": pred, "log_loss": log_loss}



def otis_aipw_superlearner(df: pd.DataFrame, *,
                            treatment: str, outcome: str,
                            covariates: list[str],
                            n_folds: int = 5, seed: int = 123,
                            eps: float = 0.02,
                            propensity_calibration: str = "none",
                            ) -> CausalEstimate:
    """SuperLearner-stacked AIPW (cross-fitted convex stack of learners).

    Stacks four learners -- random forest, ridge, OLS/logistic, mean --
    via cross-validated convex weights minimising MSE (regression) or
    log-loss (propensity). Mirrors the OTIS-RC AIPW SuperLearner
    pipeline (notez1a.qmd lines 2028-2048: SL.glmnet, SL.xgboost,
    SL.glm, SL.mean) at a lighter sklearn-only stack. xgboost is
    optional; if importable, it is added to the stack.

    Falls back to plain AIPW if sklearn isn't available.

    References
    ----------
    van der Laan, M. J. and Polley, E. C. and Hubbard, A. E. (2007).
      Super Learner. Stat. Appl. Genet. Mol. Biol. 6(1): Article 25.
    """
    try:
        from sklearn.ensemble import (RandomForestRegressor,
                                        RandomForestClassifier)
        from sklearn.linear_model import (Ridge, LogisticRegression)
    except ImportError:
        # Fall back to plain AIPW with a note
        result = otis_aipw(df, treatment=treatment, outcome=outcome,
                            covariates=covariates, n_folds=n_folds,
                            seed=seed, eps=eps,
                            propensity_calibration=propensity_calibration)
        result.notes = ["sklearn unavailable -- fell back to plain AIPW",
                         *result.notes]
        return result

    try:
        import xgboost as xgb
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False

    data = df[[treatment, outcome] + covariates].dropna().copy()
    d = _binarise(data[treatment])
    y = data[outcome].astype(np.float64).to_numpy()
    X = _design_matrix(data, covariates)
    n = y.size
    n_treated = int(d.sum())
    p_treat = float(d.mean())

    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), n_folds)

    # Define learner factories
    def _outcome_learners():
        out = [
            ("rf", RandomForestRegressor(n_estimators=300, max_depth=5,
                                            random_state=seed, n_jobs=-1)),
            ("ridge", Ridge(alpha=1.0)),
        ]
        if HAS_XGB:
            out.append(("xgb", xgb.XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                random_state=seed, n_jobs=-1, verbosity=0)))
        return out

    def _propensity_learners():
        out = [
            ("rf", RandomForestClassifier(n_estimators=300, max_depth=5,
                                              random_state=seed, n_jobs=-1)),
            ("logit", LogisticRegression(max_iter=200, solver="lbfgs",
                                              C=1.0)),
        ]
        if HAS_XGB:
            out.append(("xgb", xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                random_state=seed, n_jobs=-1, verbosity=0,
                use_label_encoder=False, eval_metric="logloss")))
        return out

    # Cross-fit each learner separately, then stack.
    e_hat = np.empty(n)
    mu1_hat = np.empty(n)
    mu0_hat = np.empty(n)

    # For stacking weights we need per-learner out-of-fold predictions.
    out_learn = _outcome_learners()
    prop_learn = _propensity_learners()

    mu1_oof = np.full((n, len(out_learn)), np.nan)
    mu0_oof = np.full((n, len(out_learn)), np.nan)
    e_oof = np.full((n, len(prop_learn)), np.nan)

    for k in range(n_folds):
        test = folds[k]
        train = np.setdiff1d(np.arange(n), test)
        # Propensity learners
        for j, (name, model) in enumerate(prop_learn):
            try:
                m = type(model)(**model.get_params())
                m.fit(X[train], d[train])
                if hasattr(m, "predict_proba"):
                    p = m.predict_proba(X[test])
                    e_oof[test, j] = p[:, 1] if p.shape[1] >= 2 else p[:, 0]
                else:
                    e_oof[test, j] = m.predict(X[test])
            except Exception:  # noqa: BLE001
                e_oof[test, j] = float(d.mean())
        # Outcome learners (separate for D=1 and D=0)
        for j, (name, model) in enumerate(out_learn):
            for dval, mu_oof in ((1, mu1_oof), (0, mu0_oof)):
                mask = train[d[train] == dval]
                if mask.size < 5:
                    mu_oof[test, j] = (y[mask].mean()
                                        if mask.size else y[train].mean())
                    continue
                try:
                    m = type(model)(**model.get_params())
                    m.fit(X[mask], y[mask])
                    mu_oof[test, j] = m.predict(X[test])
                except Exception:  # noqa: BLE001
                    mu_oof[test, j] = (y[mask].mean()
                                        if mask.size else y[train].mean())

    # Mean predictions for "mean" baseline learner
    mu_mean = float(y.mean())
    mu1_oof = np.column_stack([mu1_oof, np.full(n, mu_mean)])
    mu0_oof = np.column_stack([mu0_oof, np.full(n, mu_mean)])
    e_oof = np.column_stack([e_oof, np.full(n, p_treat)])

    # Stack via non-negative least squares with convex constraint
    def _stack_weights(P_oof: np.ndarray, target: np.ndarray) -> np.ndarray:
        # Restrict target / pred to rows where target is observed
        mask = ~np.isnan(P_oof).any(axis=1)
        if mask.sum() < 10:
            w = np.ones(P_oof.shape[1]) / P_oof.shape[1]
            return w
        Pm = P_oof[mask]
        ym = target[mask]
        # Solve constrained NNLS via simple grid + projection (fallback)
        try:
            from scipy.optimize import nnls
            w, _ = nnls(Pm, ym)
            if w.sum() > 0:
                return w / w.sum()
        except Exception:  # noqa: BLE001
            pass
        # Fallback: equal weights
        return np.ones(Pm.shape[1]) / Pm.shape[1]

    # Restrict outcome stacking to the relevant arm
    w_mu1 = _stack_weights(mu1_oof[d == 1], y[d == 1])
    w_mu0 = _stack_weights(mu0_oof[d == 0], y[d == 0])
    w_e = _stack_weights(e_oof, d.astype(np.float64))

    mu1_hat = mu1_oof @ w_mu1
    mu0_hat = mu0_oof @ w_mu0
    e_hat = _propensity_clip(e_oof @ w_e, eps=eps)

    if propensity_calibration != "none":
        e_hat = _calibrate_propensity(e_hat, d, method=propensity_calibration)
        e_hat = _propensity_clip(e_hat, eps=eps)

    # Doubly-robust score
    psi = (mu1_hat - mu0_hat
           + d * (y - mu1_hat) / e_hat
           - (1 - d) * (y - mu0_hat) / (1 - e_hat))
    ate = float(psi.mean())
    se = float(psi.std(ddof=1) / math.sqrt(n))
    z = ate / se if se > 0 else 0.0
    pval = float(2 * (1 - sps.norm.cdf(abs(z))))

    learners_used = [name for name, _ in out_learn] + ["mean"]
    notes = [f"cross-fit folds={n_folds}",
             f"learners={','.join(learners_used)}",
             f"calibration={propensity_calibration}"]
    diag = _propensity_diagnostics(e_hat, d)
    notes.append(f"Brier={diag['brier']:.3f}")
    notes.append(f"weights mu1={[round(float(x), 2) for x in w_mu1]}")
    return CausalEstimate("SuperLearner-AIPW", ate, se, pval,
                            n, n_treated, p_treat, notes)


# ── Propensity-score matching (1:k NN with caliper) ───────────────


def otis_psm(df: pd.DataFrame, *,
              treatment: str, outcome: str,
              covariates: list[str],
              k: int = 1,
              caliper_sd: float | None = 0.2,
              with_replacement: bool = False,
              eps: float = 0.02) -> CausalEstimate:
    """1:k nearest-neighbour propensity-score matching on logit(e(X)).

    Estimates the average treatment effect on the treated (ATT) by
    pairing each treated unit with its $k$ nearest control units on
    the logit-propensity scale.  An optional caliper (default
    $0.2\\,\\sigma_{\\mathrm{logit}\\,e}$, the Austin 2011
    convention) discards treated units with no control inside the
    caliper.  When $k=1$ and ``with_replacement=False`` the SE is
    the paired-difference standard error
    $\\sqrt{\\mathrm{Var}(Y_t - Y_{c(t)}) / m}$ where $m$ is the
    number of matched pairs.

    References
    ----------
    Austin, P. C. (2011).  An introduction to propensity score
    methods for reducing the effects of confounding in observational
    studies.  Multivariate Behavioural Research 46(3): 399-424.

    Abadie, A. and Imbens, G. W. (2006).  Large sample properties of
    matching estimators for average treatment effects.  Econometrica
    74(1): 235-267.
    """
    data = df[[treatment, outcome] + covariates].dropna().copy()
    d = _binarise(data[treatment])
    y = data[outcome].astype(np.float64).to_numpy()
    X = _design_matrix(data, covariates)
    n = y.size
    n_treated = int(d.sum())
    p_treat = float(d.mean())

    # Propensity score
    beta = _logit_fit(X, d)
    eta = np.clip(X @ beta, -30, 30)
    e = _propensity_clip(1.0 / (1.0 + np.exp(-eta)), eps=eps)
    logit_e = np.log(e / (1 - e))

    sd_logit = float(logit_e.std(ddof=1))
    caliper = caliper_sd * sd_logit if caliper_sd is not None else None

    treated_idx = np.where(d == 1)[0]
    control_idx = np.where(d == 0)[0]

    if treated_idx.size == 0 or control_idx.size == 0:
        return CausalEstimate("PSM", float("nan"), float("nan"), 1.0,
                                n, n_treated, p_treat,
                                notes=["no treated or no control"])

    # Greedy 1:k NN matching on logit(e), in random order to avoid
    # systematic bias from input ordering.
    rng = np.random.default_rng(123)
    treated_order = rng.permutation(treated_idx)

    available = np.ones(control_idx.size, dtype=bool)
    matches: list[tuple[int, list[int]]] = []
    n_caliper_drop = 0
    for t in treated_order:
        # logit-distances to all controls
        dist = np.abs(logit_e[control_idx] - logit_e[t])
        if not with_replacement:
            dist = np.where(available, dist, np.inf)
        if caliper is not None:
            dist = np.where(dist <= caliper, dist, np.inf)
        order = np.argsort(dist)[:k]
        valid = order[np.isfinite(dist[order])]
        if valid.size == 0:
            n_caliper_drop += 1
            continue
        if not with_replacement:
            available[valid] = False
        matches.append((int(t), [int(control_idx[v]) for v in valid]))

    if not matches:
        return CausalEstimate("PSM", float("nan"), float("nan"), 1.0,
                                n, n_treated, p_treat,
                                notes=[f"no matches under caliper={caliper}"])

    # Per-pair difference (matched-control mean averaged over k)
    diffs = np.array([y[t] - np.mean(y[ctrls])
                       for t, ctrls in matches], dtype=np.float64)
    att = float(diffs.mean())
    se = float(diffs.std(ddof=1) / math.sqrt(diffs.size)) if diffs.size > 1 else 0.0
    z = att / se if se > 0 else 0.0
    pval = float(2 * (1 - sps.norm.cdf(abs(z))))

    notes = [f"k={k}", f"caliper={caliper_sd}*SD(logit e)" if caliper_sd
              else "no caliper",
              f"matched={len(matches)}/{n_treated}",
              "no replacement" if not with_replacement else "with replacement"]
    if n_caliper_drop:
        notes.append(f"{n_caliper_drop} treated dropped at caliper")
    return CausalEstimate("PSM", att, se, pval, n, n_treated, p_treat, notes)


# ── Balance diagnostics (standardised mean differences) ────────────


def _smd(x: np.ndarray, d: np.ndarray, w: np.ndarray | None = None) -> float:
    """Standardised mean difference of binary or continuous covariate.

    For weighted samples (``w`` is non-None), uses weighted means and
    weighted variances; for unweighted, the standard
    Cohen-style pooled-SD formula.
    """
    if w is None:
        x_t = x[d == 1]
        x_c = x[d == 0]
        if x_t.size == 0 or x_c.size == 0:
            return float("nan")
        m_t, m_c = x_t.mean(), x_c.mean()
        s_t, s_c = x_t.var(ddof=1), x_c.var(ddof=1)
    else:
        wt = w * (d == 1)
        wc = w * (d == 0)
        sw_t, sw_c = wt.sum(), wc.sum()
        if sw_t == 0 or sw_c == 0:
            return float("nan")
        m_t = (wt * x).sum() / sw_t
        m_c = (wc * x).sum() / sw_c
        s_t = (wt * (x - m_t) ** 2).sum() / sw_t
        s_c = (wc * (x - m_c) ** 2).sum() / sw_c
    pooled = math.sqrt(max((s_t + s_c) / 2, 1e-12))
    return float((m_t - m_c) / pooled)


def otis_balance(df: pd.DataFrame, *,
                  treatment: str,
                  covariates: list[str],
                  outcome: str | None = None,
                  caliper_sd: float | None = 0.2,
                  eps: float = 0.02) -> pd.DataFrame:
    """Per-covariate standardised mean differences before / after
    weighting / matching.

    Returns a DataFrame with columns
    ``covariate, smd_raw, smd_ipw, smd_psm``.  Conventionally
    $|\\mathrm{SMD}| > 0.10$ flags meaningful imbalance and
    $|\\mathrm{SMD}| > 0.25$ strong imbalance \\cite{austin2011smd}.
    """
    cols_for_dropna = [treatment] + covariates + ([outcome] if outcome else [])
    data = df[cols_for_dropna].dropna().copy()
    d = _binarise(data[treatment])

    # Build the design matrix and recover its column names so we can
    # report SMDs at the dummy-column resolution.
    X_df = data[covariates].copy()
    cat_cols = X_df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if cat_cols:
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)
    X_df = X_df.astype(np.float64)
    col_names = list(X_df.columns)

    # IPW weights
    X_design = X_df.copy()
    X_design.insert(0, "_intercept", 1.0)
    Xnp = X_design.to_numpy()
    beta = _logit_fit(Xnp, d)
    eta = np.clip(Xnp @ beta, -30, 30)
    e = _propensity_clip(1.0 / (1.0 + np.exp(-eta)), eps=eps)
    w_ipw = d / e + (1 - d) / (1 - e)

    # PSM matched indices (re-use otis_psm machinery, but inline so we
    # only need to fit the propensity once)
    logit_e = np.log(e / (1 - e))
    sd_logit = float(logit_e.std(ddof=1))
    caliper = caliper_sd * sd_logit if caliper_sd is not None else None
    rng = np.random.default_rng(123)
    treated_idx = np.where(d == 1)[0]
    control_idx = np.where(d == 0)[0]
    treated_order = rng.permutation(treated_idx)
    available = np.ones(control_idx.size, dtype=bool)
    matched_t, matched_c = [], []
    for t in treated_order:
        dist = np.abs(logit_e[control_idx] - logit_e[t])
        dist = np.where(available, dist, np.inf)
        if caliper is not None:
            dist = np.where(dist <= caliper, dist, np.inf)
        nearest = int(np.argmin(dist))
        if not np.isfinite(dist[nearest]):
            continue
        available[nearest] = False
        matched_t.append(int(t))
        matched_c.append(int(control_idx[nearest]))

    rows = []
    for j, name in enumerate(col_names):
        x = X_df.iloc[:, j].to_numpy()
        smd_raw = _smd(x, d)
        smd_ipw = _smd(x, d, w=w_ipw)
        if matched_t:
            mask = np.zeros_like(d, dtype=bool)
            mask[matched_t] = True
            mask[matched_c] = True
            smd_psm = _smd(x[mask], d[mask])
        else:
            smd_psm = float("nan")
        rows.append({"covariate": name,
                      "smd_raw": round(smd_raw, 4),
                      "smd_ipw": round(smd_ipw, 4),
                      "smd_psm": round(smd_psm, 4)})
    return pd.DataFrame(rows)


# ── Overlap (common-support) diagnostics ───────────────────────────


def otis_overlap(df: pd.DataFrame, *,
                  treatment: str,
                  covariates: list[str],
                  eps: float = 0.02) -> dict:
    """Propensity-score overlap diagnostics.

    Returns a dict with min/max propensity per group, fraction inside
    the trimming interval $[\\varepsilon, 1-\\varepsilon]$,
    Hellinger distance between treated/control propensity densities,
    and the count of clipped propensities.
    """
    data = df[[treatment] + covariates].dropna().copy()
    d = _binarise(data[treatment])
    X = _design_matrix(data, covariates)
    beta = _logit_fit(X, d)
    eta = np.clip(X @ beta, -30, 30)
    e = 1.0 / (1.0 + np.exp(-eta))
    e_t = e[d == 1]
    e_c = e[d == 0]

    # Histogram-based Hellinger
    bins = np.linspace(0, 1, 21)
    h_t, _ = np.histogram(e_t, bins=bins, density=True)
    h_c, _ = np.histogram(e_c, bins=bins, density=True)
    bw = bins[1] - bins[0]
    hellinger = float(np.sqrt(0.5 * np.sum(
        (np.sqrt(h_t * bw) - np.sqrt(h_c * bw)) ** 2)))

    return {
        "min_e_treated": float(e_t.min()),
        "max_e_treated": float(e_t.max()),
        "min_e_control": float(e_c.min()),
        "max_e_control": float(e_c.max()),
        "frac_in_trim": float(((e >= eps) & (e <= 1 - eps)).mean()),
        "frac_clipped": float(((e < eps) | (e > 1 - eps)).mean()),
        "hellinger_distance": hellinger,
        "n_treated": int(d.sum()),
        "n_control": int((1 - d).sum()),
    }


# ── Cluster-robust standard errors ──────────────────────────────────


def _cluster_se(scores: np.ndarray, cluster: np.ndarray) -> float:
    """Liang-Zeger cluster-robust SE for the influence-function mean.

    For a 1-D `scores` vector (one observation per row) and `cluster`
    cluster identifiers (any hashable dtype), returns the
    cluster-robust standard error of `np.mean(scores)`:

        Var(\\bar\\psi) = (1/n^2) \\sum_g (\\sum_{i\\in g} \\psi_i)^2

    which is the standard one-way cluster sandwich
    \\cite{LiangZeger1986}.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    cluster = np.asarray(cluster).ravel()
    if scores.size != cluster.size:
        raise ValueError(f"scores ({scores.size}) and cluster "
                          f"({cluster.size}) length mismatch")
    n = scores.size
    # Group sum of scores per cluster
    df_cl = pd.DataFrame({"s": scores, "g": cluster})
    grp = df_cl.groupby("g", sort=False)["s"].sum().to_numpy()
    var = float(np.sum(grp ** 2)) / (n ** 2)
    return math.sqrt(max(var, 0.0))


def _multiway_cluster_se(scores: np.ndarray,
                          clusters: list[np.ndarray]) -> float:
    """Cameron-Gelbach-Miller multi-way cluster-robust SE.

    For a list of cluster vectors (e.g. [id_array, region_array]),
    returns the multi-way SE via inclusion-exclusion:

        V_{A,B}      = V_A + V_B - V_{A\\cap B}
        V_{A,B,C}    = V_A+V_B+V_C - V_{AB} - V_{AC} - V_{BC} + V_{ABC}

    \\cite{CameronGelbachMiller2011}.  We implement up to 2-way here;
    higher orders fall back to one-way on the first cluster.
    """
    if len(clusters) == 1:
        return _cluster_se(scores, clusters[0])
    if len(clusters) == 2:
        a, b = clusters
        intersect = np.array([f"{x}|{y}" for x, y in zip(a, b)])
        v_a = _cluster_se(scores, a) ** 2
        v_b = _cluster_se(scores, b) ** 2
        v_ab = _cluster_se(scores, intersect) ** 2
        return math.sqrt(max(v_a + v_b - v_ab, 0.0))
    # 3+ way: fall back to first axis with a warning emit
    warnings.warn(f"multiway clustering with {len(clusters)} dims "
                   "not implemented; using first axis only")
    return _cluster_se(scores, clusters[0])


# ── IRM-DML (Interactive Regression Model) ─────────────────────────


def otis_irm_dml(df: pd.DataFrame, *,
                  treatment: str, outcome: str,
                  covariates: list[str],
                  cluster_cols: list[str] | str | None = None,
                  n_folds: int = 3, seed: int = 123,
                  eps: float = 0.02,
                  ml_outcome: str = "rf",
                  ml_propensity: str = "rf",
                  match_first: bool = False,
                  match_caliper_sd: float | None = 0.2) -> dict:
    """Interactive-Regression-Model DML, matching DoubleML in R.

    Computes both the ATE (population average treatment effect) and
    ATTE (average treatment effect on the treated) via the
    doubly-robust influence function

        \\psi^{ATE}(O; \\eta)
            = \\mu_1(X) - \\mu_0(X)
              + D(Y - \\mu_1(X)) / e(X)
              - (1-D)(Y - \\mu_0(X)) / (1 - e(X)),

        \\psi^{ATTE}(O; \\eta)
            = \\frac{D(Y - \\mu_0(X)) - e(X)(1-D)(Y - \\mu_0(X)) /
                       (1 - e(X))}{\\bar D},

    with cross-fitted random-forest nuisance models.  Standard errors
    can be cluster-robust on one or two cluster axes
    (\\cite{CameronGelbachMiller2011,LiangZeger1986}).

    Returns a dict with both estimands, their SEs, and 95% CIs.

    Parameters
    ----------
    cluster_cols : str | list[str] | None
        Column names of cluster axes.  ``None`` ⇒ heteroskedasticity-
        consistent (no clustering).  ``"id"`` or ``["id"]`` ⇒ one-way
        cluster on id.  ``["id", "rc"]`` ⇒ two-way Cameron-Gelbach-Miller.
    match_first : bool
        If True, first run 1:1 nearest-neighbour propensity-score
        matching on logit(ê(X)) with caliper ``match_caliper_sd``×
        SD(logit(ê)), then fit IRM-DML on the matched subset only.
        Mirrors the MatchIt-then-DML pipeline in notez1a.qmd.
    """
    _cl_for_select = ([cluster_cols] if isinstance(cluster_cols, str)
                        else list(cluster_cols or []))
    # Dedupe column selection: cluster cols may overlap covariates; pandas
    # would otherwise return a duplicated column DataFrame on lookup.
    _cols = list(dict.fromkeys([treatment, outcome, *covariates,
                                  *_cl_for_select]))
    data = df[_cols].dropna().copy()

    if match_first:
        # Pre-match on the full data, then keep only the matched rows.
        d_all = _binarise(data[treatment])
        X_all = _design_matrix(data, covariates)
        beta = _logit_fit(X_all, d_all)
        eta = np.clip(X_all @ beta, -30, 30)
        e_all = _propensity_clip(1.0 / (1.0 + np.exp(-eta)), eps=eps)
        logit_e = np.log(e_all / (1 - e_all))
        sd_logit = float(logit_e.std(ddof=1))
        caliper = (match_caliper_sd * sd_logit
                    if match_caliper_sd is not None else None)
        rng_m = np.random.default_rng(seed + 7)
        treated_idx = np.where(d_all == 1)[0]
        control_idx = np.where(d_all == 0)[0]
        treated_order = rng_m.permutation(treated_idx)
        available = np.ones(control_idx.size, dtype=bool)
        kept = []
        for t in treated_order:
            dist = np.abs(logit_e[control_idx] - logit_e[t])
            dist = np.where(available, dist, np.inf)
            if caliper is not None:
                dist = np.where(dist <= caliper, dist, np.inf)
            nearest = int(np.argmin(dist))
            if not np.isfinite(dist[nearest]):
                continue
            available[nearest] = False
            kept.append(int(t))
            kept.append(int(control_idx[nearest]))
        if not kept:
            raise RuntimeError(
                "match_first: no treated unit had a control inside the caliper")
        data = data.iloc[sorted(set(kept))].reset_index(drop=True)

    d = _binarise(data[treatment]).astype(np.float64)
    y = data[outcome].astype(np.float64).to_numpy()
    X = _design_matrix(data, covariates)
    n = y.size
    n_treated = int(d.sum())
    p_treat = float(d.mean())

    if ml_outcome == "rf" or ml_propensity == "rf":
        try:
            from sklearn.ensemble import (RandomForestRegressor,
                                            RandomForestClassifier)
            HAS_RF = True
        except ImportError:
            HAS_RF = False
    else:
        HAS_RF = False

    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), n_folds)
    e_hat = np.empty(n)
    mu1_hat = np.empty(n)
    mu0_hat = np.empty(n)

    for k in range(n_folds):
        test = folds[k]
        train = np.setdiff1d(np.arange(n), test)
        # Propensity
        if HAS_RF and ml_propensity == "rf":
            clf = RandomForestClassifier(
                n_estimators=500, max_depth=5,
                random_state=seed + k, n_jobs=-1)
            clf.fit(X[train], d[train])
            e_hat[test] = clf.predict_proba(X[test])[:, 1]
        else:
            beta_p = _logit_fit(X[train], d[train])
            eta = np.clip(X[test] @ beta_p, -30, 30)
            e_hat[test] = 1.0 / (1.0 + np.exp(-eta))
        e_hat[test] = _propensity_clip(e_hat[test], eps=eps)
        # Outcome models for D=1 and D=0
        for dval, mu in ((1, mu1_hat), (0, mu0_hat)):
            mask = train[d[train] == dval]
            if mask.size < X.shape[1] + 2:
                mu[test] = (y[mask].mean()
                             if mask.size else y[train].mean())
                continue
            if HAS_RF and ml_outcome == "rf":
                reg = RandomForestRegressor(
                    n_estimators=500, max_depth=5,
                    random_state=seed + k * 13, n_jobs=-1)
                reg.fit(X[mask], y[mask])
                mu[test] = reg.predict(X[test])
            else:
                beta_y, *_ = np.linalg.lstsq(X[mask], y[mask],
                                                rcond=None)
                mu[test] = X[test] @ beta_y

    # ATE
    psi_ate = (mu1_hat - mu0_hat
                + d * (y - mu1_hat) / e_hat
                - (1 - d) * (y - mu0_hat) / (1 - e_hat))
    ate = float(psi_ate.mean())

    # ATTE -- efficient influence function for E[Y(1)-Y(0)|D=1]
    p_d = max(p_treat, 1e-9)
    psi_atte = (d * (y - mu0_hat)
                  - e_hat * (1 - d) * (y - mu0_hat) / (1 - e_hat)) / p_d
    atte = float(psi_atte.mean())

    # ATC -- efficient influence function for E[Y(1)-Y(0)|D=0]
    p_d0 = max(1.0 - p_treat, 1e-9)
    psi_atc = ((1 - d) * (mu1_hat - mu0_hat)
                + d * ((1 - e_hat) / e_hat) * (y - mu1_hat)
                - (1 - d) * (y - mu0_hat)) / p_d0
    atc = float(psi_atc.mean())

    # Variance -- cluster-robust if requested
    if cluster_cols is None:
        se_ate = float(psi_ate.std(ddof=1) / math.sqrt(n))
        se_atte = float(psi_atte.std(ddof=1) / math.sqrt(n))
        se_atc = float(psi_atc.std(ddof=1) / math.sqrt(n))
        se_kind = "iid"
    else:
        cl_list = ([cluster_cols] if isinstance(cluster_cols, str)
                    else list(cluster_cols))
        cluster_arrs = [data[c].to_numpy() for c in cl_list]
        # Center scores so their mean equals the estimate before SE
        se_ate = _multiway_cluster_se(psi_ate - ate, cluster_arrs)
        se_atte = _multiway_cluster_se(psi_atte - atte, cluster_arrs)
        se_atc = _multiway_cluster_se(psi_atc - atc, cluster_arrs)
        se_kind = (f"cluster:{cl_list[0]}" if len(cl_list) == 1
                    else "cluster:" + "+".join(cl_list))

    z_ate = ate / se_ate if se_ate > 0 else 0.0
    z_atte = atte / se_atte if se_atte > 0 else 0.0
    z_atc = atc / se_atc if se_atc > 0 else 0.0
    p_ate = float(2 * (1 - sps.norm.cdf(abs(z_ate))))
    p_atte = float(2 * (1 - sps.norm.cdf(abs(z_atte))))
    p_atc = float(2 * (1 - sps.norm.cdf(abs(z_atc))))
    return {
        "ate": ate, "ate_se": se_ate, "ate_pval": p_ate,
        "ate_ci95": (ate - 1.96 * se_ate, ate + 1.96 * se_ate),
        "atte": atte, "atte_se": se_atte, "atte_pval": p_atte,
        "atte_ci95": (atte - 1.96 * se_atte, atte + 1.96 * se_atte),
        "atc": atc, "atc_se": se_atc, "atc_pval": p_atc,
        "atc_ci95": (atc - 1.96 * se_atc, atc + 1.96 * se_atc),
        "n": n, "n_treated": n_treated, "p_treat": p_treat,
        "se_kind": se_kind,
        "ml_outcome": ml_outcome if HAS_RF else "ols",
        "ml_propensity": ml_propensity if HAS_RF else "logit",
    }


# ── Per-year orchestrator ──────────────────────────────────────────


def otis_per_year_irm_dml(df: pd.DataFrame, *,
                            treatment: str, outcome: str,
                            covariates: list[str],
                            year_col: str = "EndFiscalYear",
                            cluster_cols: list[str] | str | None = None,
                            n_folds: int = 3, seed: int = 123,
                            full_battery: bool = False,
                            propensity_calibration: str = "none") -> dict:
    """Fit IRM-DML separately on each fiscal year.

    Returns a dict keyed by year value.  When ``full_battery=False``
    (default), each year's value is the standard ``otis_irm_dml``
    output (mirrors Ruhela's ``res_by_year`` in
    ``correctional_stats_report1z.RData``).

    When ``full_battery=True``, each year's value is a dict
    ``{estimator_name: CausalEstimate-or-irm-dict}`` with all 10
    Ruhela-formulation estimators (IPW, AIPW, g-computation, PSM 1:1
    NN, PSM subclass, IRM-DML, PSM->IRM-DML match_first, ATC-AIPW,
    PLR, SuperLearner-AIPW). Roughly 7× the per-year runtime.
    """
    out: dict = {}
    for yr, sub in df.groupby(year_col):
        if not full_battery:
            try:
                r = otis_irm_dml(sub, treatment=treatment, outcome=outcome,
                                  covariates=covariates,
                                  cluster_cols=cluster_cols,
                                  n_folds=n_folds, seed=seed)
                out[str(yr)] = r
            except Exception as exc:  # noqa: BLE001
                out[str(yr)] = {"error": str(exc)[:120]}
            continue

        # Full battery: run every estimator on this year's subset
        year_results: dict = {"year": int(yr) if str(yr).isdigit()
                                else str(yr),
                                "n": int(sub.shape[0])}
        for label, runner, kwargs in [
            ("ipw", otis_ipw,
                {"propensity_calibration": propensity_calibration}),
            ("aipw", otis_aipw,
                {"propensity_calibration": propensity_calibration,
                 "n_folds": n_folds}),
            ("gcomp", otis_gcomputation, {"n_bootstrap": 100}),
            ("psm_nn", otis_psm, {"k": 1}),
            ("psm_subclass", otis_psm_subclass, {"n_strata": 5}),
            ("atc", otis_atc, {"n_folds": n_folds}),
            ("plr", otis_plr, {"n_folds": n_folds}),
            ("superlearner", otis_aipw_superlearner,
                {"propensity_calibration": propensity_calibration,
                 "n_folds": n_folds}),
        ]:
            try:
                est = runner(sub, treatment=treatment, outcome=outcome,
                              covariates=covariates, **kwargs)
                year_results[label] = {
                    "estimator": est.estimator,
                    "ate": float(est.ate),
                    "se": float(est.ate_se),
                    "p": float(est.ate_pval),
                    "ci95": list(est.ate_ci95),
                    "n": int(est.n),
                }
            except Exception as exc:  # noqa: BLE001
                year_results[label] = {"error": str(exc)[:120]}
        # IRM-DML (cluster-robust if requested) -- separately because
        # it returns ATE+ATTE+ATC dict, not CausalEstimate
        try:
            irm = otis_irm_dml(sub, treatment=treatment, outcome=outcome,
                                covariates=covariates,
                                cluster_cols=cluster_cols,
                                n_folds=n_folds, seed=seed)
            year_results["irm_dml"] = {
                "ate": float(irm["ate"]), "ate_se": float(irm["ate_se"]),
                "atte": float(irm["atte"]), "atte_se": float(irm["atte_se"]),
                "atc": float(irm.get("atc", float("nan")))
                if irm.get("atc") is not None else None,
                "atc_se": float(irm.get("atc_se", float("nan")))
                if irm.get("atc_se") is not None else None,
                "se_kind": irm["se_kind"],
                "n": int(irm["n"]),
            }
        except Exception as exc:  # noqa: BLE001
            year_results["irm_dml"] = {"error": str(exc)[:120]}
        # match_first (PSM->IRM-DML)
        try:
            mf = otis_irm_dml(sub, treatment=treatment, outcome=outcome,
                                covariates=covariates,
                                cluster_cols=cluster_cols,
                                match_first=True,
                                n_folds=n_folds, seed=seed)
            year_results["match_first"] = {
                "ate": float(mf["ate"]), "ate_se": float(mf["ate_se"]),
                "n": int(mf["n"]),
            }
        except Exception as exc:  # noqa: BLE001
            year_results["match_first"] = {"error": str(exc)[:120]}
        out[str(yr)] = year_results
    return out


# ── Ruhela's primary T->Y pair: ac >= 2 -> vm (Goffmanian classification) ──


def make_pair_alert_to_volatility_ruhela(df: pd.DataFrame
                                          ) -> tuple[pd.DataFrame, str, str, list[str]]:
    """Ruhela's primary causal contrast: high alert complexity (ac ≥ 2)
    causes more inter-region transfers (vm).

    **This is "Ruhela's equation" for alert complexity and volatility**
    -- the 8-state combo encoding documented in
    ``/path/to/workspace/OTIS-RC/notez1a.qmd`` and used to
    produce the published ``res_pool`` / ``res_by_year`` / ``res_all``
    estimates in ``correctional_stats_report1z.RData``.  Sister
    function :func:`make_pair_alert_to_volatility_naive` is
    available as a robustness alternative; both are run side by
    side by :func:`make_pair_alert_to_volatility_all`.

    Encoding
    --------
    For each placement-row, the three binary alerts
    (MentalHealth_Alert, SuicideRisk_Alert, SuicideWatch_Alert)
    encode one of $2^3 = 8$ states:

        a1  =  MH  & ¬SR & ¬SW       (mental-health only)
        a2  = ¬MH  &  SR & ¬SW       (suicide-risk only)
        a3  = ¬MH  & ¬SR &  SW       (suicide-watch only -- empirically empty)
        a4  =  MH  &  SR & ¬SW       (MH+SR)
        a5  = ¬MH  &  SR &  SW       (SR+SW)
        a6  =  MH  & ¬SR &  SW       (MH+SW -- empirically empty)
        a7  =  MH  &  SR &  SW       (all three)
        a8  = ¬MH  & ¬SR & ¬SW       (no alerts)

    Per (UniqueIndividual_ID, EndFiscalYear), the *alert-state
    complexity* `ac` is the number of distinct combos with positive
    support across the year's placement-rows for that person.  This
    differs from the naïve "max simultaneous flags" -- a person who
    had three a4 placements has ac = 1 (one distinct combo), not
    ac = 2 (two simultaneous flags).  Treatment T = 1 iff ac ≥ 2.

    Outcome
    -------
    `vm` (regional volatility moves, count) sums two indicators
    across the person-year placement sequence:
      (i) regA ≠ regB within the same row (the person's
          original-vs-most-recent regions differ in this placement);
      (ii) regA ≠ regA(prev row) across rows (the person's
           original-region label changed across consecutive placements).

    This is a Poisson-style count outcome on $\\{0, 1, 2, \\ldots\\}$.

    Returns
    -------
    (data, T, Y, covariates) where:
        T = "T_high_ac"     binary, 1 iff ac ≥ 2
        Y = "Y_vm_count"    integer count, regional transfer count
        covariates = ["Gender", "Age_Category", "EndFiscalYear"]
    """
    base = df[["UniqueIndividual_ID", "EndFiscalYear", "Gender",
                 "Age_Category", "Region_AtTimeOfPlacement",
                 "Region_MostRecentPlacement",
                 "MentalHealth_Alert", "SuicideRisk_Alert",
                 "SuicideWatch_Alert"]].dropna().copy()
    a_mh = _binarise(base["MentalHealth_Alert"])
    a_sr = _binarise(base["SuicideRisk_Alert"])
    a_sw = _binarise(base["SuicideWatch_Alert"])
    # 8-state combo encoding -- bitfield (MH * 4 + SR * 2 + SW * 1)
    base["combo"] = (a_mh * 4 + a_sr * 2 + a_sw * 1).astype(int)
    # Per-row indicator of within-row region change
    base["regA"] = base["Region_AtTimeOfPlacement"].astype(str)
    base["regB"] = base["Region_MostRecentPlacement"].astype(str)
    base["vm_within_row"] = (base["regA"] != base["regB"]).astype(int)
    # Sort by (id, year) for the across-row shift
    base = base.sort_values(["UniqueIndividual_ID", "EndFiscalYear"])
    # vm across rows: regA != regA(prev), per (id, year) group
    base["regA_prev"] = (base.groupby(
        ["UniqueIndividual_ID", "EndFiscalYear"])["regA"].shift(1))
    base["vm_across_row"] = ((base["regA"] != base["regA_prev"])
                                & base["regA_prev"].notna()).astype(int)
    base["vm_row"] = base["vm_within_row"] + base["vm_across_row"]

    # Per (id, year) aggregation
    def _ac_distinct_combos(s):
        return int(s.nunique())

    py = (base.groupby(["UniqueIndividual_ID", "EndFiscalYear"])
                .agg(ac=("combo", _ac_distinct_combos),
                      vm=("vm_row", "sum"),
                      Gender=("Gender", "first"),
                      Age_Category=("Age_Category", "first"),
                      regA=("regA", "first"),
                      regB=("regB", "first"))
                .reset_index())
    py["T_high_ac"] = (py["ac"] >= 2).astype(int)
    py["Y_vm_count"] = py["vm"].astype(int)
    return (py, "T_high_ac", "Y_vm_count",
             ["Gender", "Age_Category", "EndFiscalYear"])


def make_pair_alert_to_volatility_naive(df: pd.DataFrame
                                         ) -> tuple[pd.DataFrame, str, str, list[str]]:
    """Naïve (simpler) alternative to :func:`make_pair_alert_to_volatility_ruhela`.

    Treatment ac:
        per (id, year), the *maximum* number of simultaneous alert
        flags across the year's placement rows
        (i.e.\\ ``max(MH + SR + SW)`` per row).
        T = 1 iff this maximum is ≥ 2.

    Outcome vm (binary):
        per (id, year), 1 iff any placement row has
        ``Region_AtTimeOfPlacement ≠ Region_MostRecentPlacement``.

    This was the original morie formulation before we audited
    against ``OTIS-RC/notez1a.qmd``; it produces a different
    treatment marginal (~24\\% treated vs ~14\\% under Ruhela's
    8-state encoding) and a binary rather than count outcome.  We
    keep it for robustness comparison: if both formulations agree
    on sign and rough magnitude, the substantive conclusion is
    invariant to the operationalisation.
    """
    base = df[["UniqueIndividual_ID", "EndFiscalYear", "Gender",
                 "Age_Category", "Region_AtTimeOfPlacement",
                 "Region_MostRecentPlacement",
                 "MentalHealth_Alert", "SuicideRisk_Alert",
                 "SuicideWatch_Alert"]].dropna().copy()
    base["alert_sum_row"] = (_binarise(base["MentalHealth_Alert"])
                                + _binarise(base["SuicideRisk_Alert"])
                                + _binarise(base["SuicideWatch_Alert"]))
    base["vm_row_binary"] = (base["Region_AtTimeOfPlacement"]
                                != base["Region_MostRecentPlacement"]).astype(int)
    py = (base.groupby(["UniqueIndividual_ID", "EndFiscalYear"])
                .agg(ac_naive=("alert_sum_row", "max"),
                      vm_any=("vm_row_binary", "max"),
                      Gender=("Gender", "first"),
                      Age_Category=("Age_Category", "first"))
                .reset_index())
    py["T_high_ac"] = (py["ac_naive"] >= 2).astype(int)
    py["Y_vm_any"] = py["vm_any"].astype(int)
    return (py, "T_high_ac", "Y_vm_any",
             ["Gender", "Age_Category", "EndFiscalYear"])


def make_pair_alert_to_volatility_all(df: pd.DataFrame) -> dict:
    """Run both Ruhela and naïve alert->volatility formulations.

    Returns a dict ``{"ruhela": (...), "naive": (...)}`` where each
    value is a 4-tuple ``(person_year_data, T, Y, covariates)``
    suitable for handing to :func:`otis_irm_dml`.
    """
    return {
        "ruhela": make_pair_alert_to_volatility_ruhela(df),
        "naive": make_pair_alert_to_volatility_naive(df),
    }


# Backwards-compatible alias -- points to Ruhela's formulation, the
# faithful replication of OTIS-RC/notez1a.qmd.
make_pair_alert_to_volatility = make_pair_alert_to_volatility_ruhela


# ── DML wrapper to match the IPW / AIPW signature ──────────────────


def otis_dml(df: pd.DataFrame, *,
              treatment: str, outcome: str,
              covariates: list[str],
              n_folds: int = 5, seed: int = 123) -> CausalEstimate:
    """Cross-fitted DML PLR (partialling-out) wrapper.

    Reproduces :func:`morie.otis.otdml` but exposes the same
    :class:`CausalEstimate` shape for grid comparison.
    """
    data = df[[treatment, outcome] + covariates].dropna().copy()
    d = _binarise(data[treatment]).astype(np.float64)
    y = data[outcome].astype(np.float64).to_numpy()
    X = _design_matrix(data, covariates)
    n = y.size
    n_treated = int(d.sum())
    p_treat = float(d.mean())

    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), n_folds)
    y_res = np.empty(n)
    d_res = np.empty(n)
    for k in range(n_folds):
        test = folds[k]
        train = np.setdiff1d(np.arange(n), test)
        beta_y, *_ = np.linalg.lstsq(X[train], y[train], rcond=None)
        y_res[test] = y[test] - X[test] @ beta_y
        beta_d, *_ = np.linalg.lstsq(X[train], d[train], rcond=None)
        d_res[test] = d[test] - X[test] @ beta_d

    # ATE via partialled-out regression
    num = float((d_res * y_res).sum())
    den = float((d_res ** 2).sum())
    ate = num / den if den > 0 else 0.0
    resid = y_res - d_res * ate
    meat = float(np.mean((d_res ** 2) * (resid ** 2)))
    bread = float(np.mean(d_res ** 2))
    se = float(math.sqrt(meat / (bread ** 2 * n))) if bread > 0 else 0.0
    z = ate / se if se > 0 else 0.0
    pval = float(2 * (1 - sps.norm.cdf(abs(z))))
    return CausalEstimate("DML", ate, se, pval, n, n_treated, p_treat,
                            notes=[f"cross-fit folds={n_folds}"])


# ── Treatment / outcome constructors for the three pairs ───────────


def _ensure_b01(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is not None:
        return df
    from .otis_churn import _load
    return _load("b01")


def _ensure_a01(df: pd.DataFrame | None,
                 *, auto_download: bool = True) -> pd.DataFrame:
    """Return Ruhela's canonical a01 (Restrictive Confinement Detailed)
    DataFrame, in the column-name convention used by
    ``morie.otis_causal``.

    Resolution order:
      1. ``df`` argument if non-None (caller supplied);
      2. local CSV at
         ``data/datasets/OTIS/a01_restrictive_confinement_detailed_dataset.csv``;
      3. live CKAN download via
         :func:`morie.otis_datasets.download_otis_dataset` (when
         ``auto_download=True``).

    Column names from CKAN are returned verbatim (no rename), so they
    match the rest of the otis_causal pipeline.
    """
    if df is not None:
        return df
    from . import otis_datasets as ods
    csv = ods.OTIS_DATA_DIR / "a01_restrictive_confinement_detailed_dataset.csv"
    if not csv.exists() and auto_download:
        csv = ods.download_otis_dataset("a01")
    if not csv.exists():
        raise FileNotFoundError(
            f"a01 CSV not found at {csv}; pass `df=` or set auto_download=True")
    return pd.read_csv(csv)


def make_pair_a(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str, list[str]]:
    """Pair (a): MentalHealth_Alert -> SuicideRisk_Alert (binary->binary).

    The clinical-alert chain: do mental-health flags causally elevate
    subsequent suicide-risk-alert occurrence, conditional on
    demographics and region?
    """
    base = df[["UniqueIndividual_ID", "EndFiscalYear", "Gender",
                 "Age_Category", "Region_AtTimeOfPlacement",
                 "Region_MostRecentPlacement", "MentalHealth_Alert",
                 "SuicideRisk_Alert"]].dropna().copy()
    base["T_a"] = _binarise(base["MentalHealth_Alert"])
    base["Y_a"] = _binarise(base["SuicideRisk_Alert"])
    covariates = ["Gender", "Age_Category",
                   "Region_AtTimeOfPlacement",
                   "Region_MostRecentPlacement"]
    return base, "T_a", "Y_a", covariates


def make_pair_b(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str, list[str]]:
    """Pair (b): HighAlertComplexity -> AnyReadmission.

    Treatment T_b = 1 iff at least 2 of {MentalHealth, SuicideRisk,
    SuicideWatch} alerts are simultaneously active in the year.
    Outcome Y_b = 1 iff this individual has ≥2 placements (proxy for
    any future readmission).
    """
    base = df[["UniqueIndividual_ID", "EndFiscalYear", "Gender",
                 "Age_Category", "Region_AtTimeOfPlacement",
                 "Region_MostRecentPlacement",
                 "MentalHealth_Alert", "SuicideRisk_Alert",
                 "SuicideWatch_Alert", "Number_Of_Placements"]].dropna().copy()
    a1 = _binarise(base["MentalHealth_Alert"])
    a2 = _binarise(base["SuicideRisk_Alert"])
    a3 = _binarise(base["SuicideWatch_Alert"])
    base["T_b"] = ((a1 + a2 + a3) >= 2).astype(int)
    base["Y_b"] = (pd.to_numeric(base["Number_Of_Placements"],
                                   errors="coerce")
                     .fillna(0)
                     .astype(int) >= 2).astype(int)
    covariates = ["Gender", "Age_Category",
                   "Region_AtTimeOfPlacement",
                   "Region_MostRecentPlacement"]
    return base, "T_b", "Y_b", covariates


def make_pair_c(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str, list[str]]:
    """Pair (c): RegionalVolatility -> SegregationDays.

    Treatment T_c = 1 iff Region_AtTimeOfPlacement ≠
    Region_MostRecentPlacement (the inter-region transfer flag).
    Outcome Y_c = NumberConsecutiveDays_Segregation (continuous,
    truncated at the 99th percentile to dampen the long tail).
    """
    base = df[["UniqueIndividual_ID", "EndFiscalYear", "Gender",
                 "Age_Category", "Region_AtTimeOfPlacement",
                 "Region_MostRecentPlacement",
                 "MentalHealth_Alert",
                 "NumberConsecutiveDays_Segregation"]].dropna().copy()
    base["T_c"] = (base["Region_AtTimeOfPlacement"]
                     != base["Region_MostRecentPlacement"]).astype(int)
    y = pd.to_numeric(base["NumberConsecutiveDays_Segregation"],
                       errors="coerce").fillna(0).astype(float)
    p99 = float(y.quantile(0.99))
    base["Y_c"] = y.clip(upper=p99)
    covariates = ["Gender", "Age_Category", "MentalHealth_Alert"]
    return base, "T_c", "Y_c", covariates


# ── The 3 × 3 grid ─────────────────────────────────────────────────


def make_pair_alert_to_volatility_a01(df: pd.DataFrame | None = None
                                        ) -> tuple[pd.DataFrame, str, str, list[str]]:
    """Same as :func:`make_pair_alert_to_volatility_ruhela` but
    auto-loads a01 (Restrictive Confinement Detailed) instead of
    b01 (Segregation Detailed) if ``df`` is None.

    a01 is the file the published `res_pool` / `res_by_year` /
    `res_all` are computed on; using it in our pipeline closes the
    "different bundled snapshot" gap.
    """
    df = _ensure_a01(df)
    return make_pair_alert_to_volatility_ruhela(df)


def otis_causal_grid(df: pd.DataFrame | None = None,
                       *,
                       seed: int = 123) -> RichResult:
    """Run IPW / AIPW / DML on all three canonical (T, Y) pairs.

    Returns a :class:`RichResult` with a 9-row payload table -- three
    estimators × three pairs -- plus per-row standard errors,
    p-values, and 95% confidence intervals.
    """
    df = _ensure_b01(df)

    pairs = {
        "(a) MentalHealth -> SuicideRisk":
            make_pair_a(df),
        "(b) HighAlertComplexity -> AnyReadmission":
            make_pair_b(df),
        "(c) RegionalVolatility -> SegregationDays":
            make_pair_c(df),
    }

    rows = []
    for label, (data, T, Y, covs) in pairs.items():
        if data[T].sum() == 0 or data[T].sum() == data.shape[0]:
            warnings.warn(f"{label}: degenerate treatment, skipping")
            continue
        for fn, kind in ((otis_ipw, "IPW"),
                           (otis_aipw, "AIPW"),
                           (otis_dml, "DML"),
                           (otis_psm, "PSM")):
            try:
                est = fn(data, treatment=T, outcome=Y,
                          covariates=covs, seed=seed) \
                       if "seed" in fn.__code__.co_varnames else \
                       fn(data, treatment=T, outcome=Y, covariates=covs)
            except Exception as exc:  # noqa: BLE001
                rows.append({"pair": label, "estimator": kind,
                              "error": str(exc)[:80]})
                continue
            rows.append({
                "pair": label,
                "estimator": kind,
                "n": est.n,
                "p_treat": round(est.p_treat, 4),
                "ate": round(est.ate, 4),
                "ate_se": round(est.ate_se, 4),
                "ate_pval": round(est.ate_pval, 4),
                "ci95_lo": round(est.ate_ci95[0], 4),
                "ci95_hi": round(est.ate_ci95[1], 4),
                "notes": "; ".join(est.notes) if est.notes else "",
            })

    summary_lines = [
        ("Pairs evaluated", len(pairs)),
        ("Estimators per pair", 3),
        ("Total rows", len([r for r in rows if "error" not in r])),
        ("Errored rows", len([r for r in rows if "error" in r])),
    ]
    interp = (
        "Three causal estimators (IPW, AIPW, DML) applied to three "
        "canonical OTIS treatment-outcome pairs.  Disagreement across "
        "estimators is informative: IPW alone is sensitive to "
        "propensity misspecification; AIPW is doubly-robust; DML "
        "with cross-fitting is robust to flexible nuisance "
        "estimation.  Concordance across all three is the strongest "
        "evidence of an identified causal effect under conditional "
        "exchangeability."
    )
    return RichResult(
        title="OTIS causal grid -- IPW / AIPW / DML × 3 (T,Y) pairs",
        summary_lines=summary_lines,
        interpretation=interp,
        payload={"rows": rows},
    )
