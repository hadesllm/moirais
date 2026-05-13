"""Reproducible replication of Laniyonu & Goff (2021) BMC Psychiatry 21(1):500.

"Measuring disparities in police use of force and injury among persons
with serious mental illness" — a Bayesian hierarchical negative-binomial
model with a *synthetic* exposure offset for persons-with-SMI (PwSMI).
The trick: there is no administrative census of who has SMI at the
tract level, so the denominator is constructed by:

  1. Fitting P(SMI | age, sex, race, income, ...) on a national survey
     (NCS-R in the paper) using only covariates that are also tabulated
     at the tract level by the ACS.
  2. Applying those coefficients to ACS tract marginals to get a
     per-tract predicted P(SMI).
  3. Multiplying by adult population for a synthetic exposure
     denominator n_{vti}.

The count model is:

    y_{vti} ~ NegBin(n_{vti} * exp(mu + alpha_v + delta_t + beta_i), phi)

where ``v`` indexes the vulnerability class (PwSMI vs non-SMI), ``t``
indexes year, and ``i`` indexes the area unit (tract or precinct).
The headline coefficient ``alpha_v`` is the log relative-risk of
police use of force against PwSMI vs non-SMI.

Headline finding the paper reports:

  - Tract-level:    RR PwSMI = 11.6x   (alpha_v = ln 11.6 ~ 2.45)
  - Precinct-level: RR PwSMI = 10.2x   (alpha_v = ln 10.2 ~ 2.32)

This module is a thin user-facing wrapper.  Heavy lifting lives in
:func:`morie.mrm_primitives.synthetic_area_exposure` — we compose it
here rather than re-implementing the SAE step.  We do NOT use
``statsmodels``, ``pymc``, or ``brms``: the negative-binomial GLM is
fit by a hand-rolled MLE via :func:`scipy.optimize.minimize`.  The
output is frequentist MLE-with-Hessian-SE, not full Bayesian
posteriors; for paper-grade credible intervals the user should pass
``return_design=True`` and feed the matrices into ``pymc`` themselves.

The SMI flag is a proxy and is biased in a known direction: officers
are more likely to MISS SMI than to over-attribute it, so the
estimated alpha_v is *conservative* (under-states true disparity).
:func:`smi_force_disparity` emits an explicit ``UserWarning`` to that
effect on every call.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize

from ..mrm_primitives.synthetic_exposure import synthetic_area_exposure


@dataclass
class SMIForceDisparityCoefficient:
    """Single coefficient with point estimate, SE, and Wald 95% CI."""
    name: str
    estimate: float
    std_error: float

    @property
    def ci_low(self) -> float:
        return self.estimate - 1.96 * self.std_error

    @property
    def ci_high(self) -> float:
        return self.estimate + 1.96 * self.std_error

    @property
    def rr(self) -> float:
        """exp(estimate) — relative risk on the multiplicative scale."""
        return float(np.exp(self.estimate))

    @property
    def rr_ci_low(self) -> float:
        return float(np.exp(self.ci_low))

    @property
    def rr_ci_high(self) -> float:
        return float(np.exp(self.ci_high))


@dataclass
class SMIForceDisparityResult:
    """Replication output for Laniyonu & Goff (2021).

    Attributes
    ----------
    alpha_v : SMIForceDisparityCoefficient
        Headline coefficient: log relative-risk of force against PwSMI
        vs non-SMI.  ``alpha_v.rr`` is the multiplicative relative risk.
    intercept : SMIForceDisparityCoefficient
        Grand intercept ``mu``.
    year_effects : list[SMIForceDisparityCoefficient]
        Year fixed effects ``delta_t`` (one per year minus baseline).
    area_random_effect_sd : float
        Estimated SD of the area-level random intercept ``beta_i``.
        Approximated by a single dispersion parameter under the
        random-intercept assumption (see note).
    dispersion : float
        Negative-binomial dispersion ``phi``.  Variance = mu + mu^2/phi.
    n_events : int
        Total force events across the panel.
    n_area_years : int
        Number of (area x year) observation rows.
    log_likelihood : float
        Final log-likelihood at the MLE.
    converged : bool
        Whether scipy.optimize reports convergence.
    exposure_summary : dict
        Summary stats on the SAE-derived offset for sanity checking.
    note : str
        Caveats / methodology notes for the user.
    """
    alpha_v: SMIForceDisparityCoefficient
    intercept: SMIForceDisparityCoefficient
    year_effects: list[SMIForceDisparityCoefficient]
    area_random_effect_sd: float
    dispersion: float
    n_events: int
    n_area_years: int
    log_likelihood: float
    converged: bool
    exposure_summary: dict = field(default_factory=dict)
    note: str = ""

    def interpret(self) -> str:
        head = (
            f"Laniyonu & Goff (2021) replication: {self.n_events:,} force "
            f"events across {self.n_area_years:,} area-year rows.  "
        )
        rr_line = (
            f"alpha_v (PwSMI log-RR) = {self.alpha_v.estimate:+.3f} "
            f"(SE {self.alpha_v.std_error:.3f}); RR = {self.alpha_v.rr:.2f}x "
            f"[95% CI {self.alpha_v.rr_ci_low:.2f}x, "
            f"{self.alpha_v.rr_ci_high:.2f}x]."
        )
        disp_line = (
            f"  NB dispersion phi = {self.dispersion:.3f}; "
            f"area random-intercept SD ~= {self.area_random_effect_sd:.3f}."
        )
        comp = (
            f"  Compare paper's headline RR = 11.6x (tract) / 10.2x (precinct)."
        )
        conv = "" if self.converged else "  WARNING: optimiser did not converge."
        return head + rr_line + disp_line + comp + conv


def smi_force_disparity(
    df: pd.DataFrame,
    *,
    survey_df: pd.DataFrame,
    survey_trait_col: str = "smi",
    survey_covariate_cols: list[str],
    area_covariate_cols: list[str] | None = None,
    force_count_col: str = "force_events",
    non_smi_count_col: str | None = None,
    geog_col: str = "tract_id",
    year_col: str = "year",
    population_col: str = "pop_18plus",
    baseline_year: int | None = None,
    include_year_fe: bool = True,
    include_area_re: bool = True,
    max_iter: int = 500,
    tol: float = 1e-6,
    return_design: bool = False,
) -> SMIForceDisparityResult:
    """Replicate Laniyonu & Goff (2021): hierarchical NB on SMI force disparity.

    Composes :func:`morie.mrm_primitives.synthetic_area_exposure` to
    build the per-area PwSMI exposure offset, then fits a negative-
    binomial GLM with year fixed effects and an area random intercept
    via hand-rolled MLE (``scipy.optimize.minimize``, no statsmodels).

    The outcome is a single relative-risk: force-against-PwSMI versus
    force-against-non-SMI.  The PwSMI events use the SAE-derived offset
    n_{vti}; the non-SMI events use ``(pop_18plus - SAE_exposure)`` as
    their offset.  The vulnerability dummy ``alpha_v`` picks up the
    log-RR.

    Parameters
    ----------
    df : pd.DataFrame
        Force-event panel, one row per (area, year).  Must contain
        ``force_count_col`` (PwSMI events), the area covariates, the
        geog id, year, and adult population.
    survey_df : pd.DataFrame
        Survey microdata (e.g., NCS-R) for fitting P(SMI | covariates).
        Passed through to the SAE primitive.
    survey_trait_col : str, default "smi"
        Binary column in survey_df flagging SMI.
    survey_covariate_cols : list[str]
        Covariates available in BOTH ``survey_df`` and ``df`` (after
        renaming via ``area_covariate_cols`` if needed).
    area_covariate_cols : list[str], optional
        Column names for the SAE covariates in ``df``.  If None, we
        assume the columns are named identically to
        ``survey_covariate_cols``.
    force_count_col : str, default "force_events"
        Count of force events against PwSMI per (area, year).
    non_smi_count_col : str or None, default None
        Count of force events against non-SMI per (area, year).  If
        None, treated as ``total_force_events - force_count_col`` and
        the user must supply a column named ``"total_force_events"``.
    geog_col : str, default "tract_id"
        Area unit identifier (tract or precinct).
    year_col : str, default "year"
        Year column.
    population_col : str, default "pop_18plus"
        Adult population per area (used for the non-SMI offset and
        passed to the SAE primitive).
    baseline_year : int, optional
        Year to drop as the reference (default = min year in df).
    include_year_fe : bool, default True
        If False, omits ``delta_t``.
    include_area_re : bool, default True
        If True, an area-level random intercept is approximated as a
        free per-area shift estimated jointly (penalised by an L2 prior
        with SD estimated from the data — a marginalised approximation,
        NOT a full Bayesian hierarchical fit).
    max_iter, tol : optimiser controls.
    return_design : bool, default False
        If True, attaches ``X``, ``y``, and ``offset`` to the result's
        ``exposure_summary`` dict so the caller can hand off to pymc.

    Returns
    -------
    SMIForceDisparityResult

    Warns
    -----
    UserWarning
        Always: about the SMI-flag proxy bias (officers miss more SMI
        than they over-attribute), so the estimated alpha_v is a
        *lower bound* on the true disparity.

    Notes
    -----
    This is a frequentist MLE approximation to the paper's Bayesian
    hierarchical model.  The "random intercept" is implemented as
    free per-area shifts with an L2 ridge penalty whose strength is
    set so the implied SD matches the residual area-level variance.
    For paper-grade credible intervals, fit the model in ``pymc`` /
    ``brms`` using the design matrix you can extract with
    ``return_design=True``.
    """
    # --- 1. Mandatory user warning about the proxy bias ----------------
    warnings.warn(
        "morie.laniyonu.smi_force_disparity: the SMI flag on force events "
        "is a proxy and is biased TOWARD THE NULL — officers are more "
        "likely to MISS SMI than to over-attribute it.  The estimated "
        "alpha_v (PwSMI relative-risk) is therefore a CONSERVATIVE LOWER "
        "BOUND on the true disparity.  See Laniyonu & Goff (2021) "
        "BMC Psych 21(1):500 §Limitations.",
        UserWarning,
        stacklevel=2,
    )

    if area_covariate_cols is None:
        area_covariate_cols = list(survey_covariate_cols)
    if len(area_covariate_cols) != len(survey_covariate_cols):
        raise ValueError(
            "area_covariate_cols must have the same length as "
            "survey_covariate_cols (one per covariate)."
        )

    # --- 2. Build the SAE exposure offset ------------------------------
    # The SAE primitive expects ONE row per area; collapse df by geog.
    area_frame = (
        df.groupby(geog_col)[area_covariate_cols + [population_col]]
        .mean(numeric_only=True)
    )
    # Rename area-frame columns to match survey covariate names so the
    # primitive sees consistent column names.
    rename_map = dict(zip(area_covariate_cols, survey_covariate_cols))
    area_frame_renamed = area_frame.rename(columns=rename_map)

    smi_exposure = synthetic_area_exposure(
        survey_df=survey_df,
        survey_trait_col=survey_trait_col,
        survey_covariate_cols=survey_covariate_cols,
        area_df=area_frame_renamed,
        area_population_col=population_col,
    )
    # smi_exposure: Series indexed by geog
    df = df.copy()
    df["_smi_exposure"] = df[geog_col].map(smi_exposure)
    df["_non_smi_exposure"] = (
        df[population_col].astype(float) - df["_smi_exposure"]
    ).clip(lower=1.0)
    df["_smi_exposure"] = df["_smi_exposure"].clip(lower=1.0)

    # --- 3. Resolve the non-SMI count column ---------------------------
    if non_smi_count_col is None:
        if "total_force_events" not in df.columns:
            raise ValueError(
                "non_smi_count_col is None but df has no "
                "'total_force_events' column to subtract from."
            )
        df["_non_smi_count"] = (
            df["total_force_events"].astype(float)
            - df[force_count_col].astype(float)
        ).clip(lower=0)
        non_smi_col_resolved = "_non_smi_count"
    else:
        non_smi_col_resolved = non_smi_count_col

    # --- 4. Stack into a long (area x year x vulnerability) frame ------
    smi_rows = df[[geog_col, year_col, force_count_col,
                   "_smi_exposure"]].copy()
    smi_rows.columns = [geog_col, year_col, "_count", "_offset"]
    smi_rows["_v"] = 1  # PwSMI

    non_smi_rows = df[[geog_col, year_col, non_smi_col_resolved,
                       "_non_smi_exposure"]].copy()
    non_smi_rows.columns = [geog_col, year_col, "_count", "_offset"]
    non_smi_rows["_v"] = 0  # non-SMI

    long_df = pd.concat([smi_rows, non_smi_rows], ignore_index=True)
    long_df = long_df.dropna(subset=["_count", "_offset"])
    long_df = long_df[long_df["_offset"] > 0]

    # --- 5. Build the design matrix ------------------------------------
    if baseline_year is None:
        baseline_year = int(long_df[year_col].min())
    years_sorted = sorted(long_df[year_col].unique().tolist())
    non_baseline_years = [y for y in years_sorted if y != baseline_year]

    design_cols = ["_intercept", "_v"]
    X_pieces = [
        np.ones(len(long_df)),
        long_df["_v"].to_numpy(dtype=float),
    ]
    if include_year_fe:
        for y in non_baseline_years:
            X_pieces.append((long_df[year_col] == y).to_numpy(dtype=float))
            design_cols.append(f"_year_{y}")

    # Area random intercept = one-hot area dummies, baseline area dropped
    area_codes = pd.Categorical(long_df[geog_col])
    area_levels = list(area_codes.categories)
    if include_area_re and len(area_levels) > 1:
        baseline_area = area_levels[0]
        non_baseline_areas = area_levels[1:]
        for a in non_baseline_areas:
            X_pieces.append(
                (long_df[geog_col] == a).to_numpy(dtype=float)
            )
            design_cols.append(f"_area_{a}")
    else:
        non_baseline_areas = []
        baseline_area = area_levels[0] if area_levels else None

    X = np.column_stack(X_pieces)
    y_count = long_df["_count"].to_numpy(dtype=float)
    offset = np.log(long_df["_offset"].to_numpy(dtype=float))

    n_obs, n_params = X.shape
    n_year_fe = len(non_baseline_years) if include_year_fe else 0
    n_area_re = len(non_baseline_areas)
    # Index ranges for parameter blocks (intercept, alpha_v, year FE, area RE)
    idx_intercept = 0
    idx_alpha_v = 1
    idx_year_start = 2
    idx_year_end = idx_year_start + n_year_fe
    idx_area_start = idx_year_end
    idx_area_end = idx_area_start + n_area_re

    # --- 6. Negative-binomial log-likelihood ---------------------------
    # NB2 parametrisation: Var(Y) = mu + mu^2 / phi.  log phi is the
    # last free parameter.  Area dummies are L2-penalised with a tau
    # parameter co-estimated from the data (penalty 0.5 * sum_a^2 / tau^2).
    # The penalty's tau is profiled out via posterior-mode of an
    # inverse-gamma weakly-informative prior on tau (alpha=2, beta=1)
    # giving tau_hat = sqrt((sum_a^2 + 2*beta) / (n_area_re + 2*alpha + 2)).
    n_free = n_params + 1  # plus log_phi
    init = np.zeros(n_free)
    # Sensible intercept seed: log(mean rate) net of offset
    with np.errstate(divide="ignore"):
        init[idx_intercept] = float(
            np.log((y_count.sum() + 1.0)
                   / (np.exp(offset).sum() + 1.0))
        )
    init[-1] = 0.0  # log_phi = 0 -> phi = 1

    def neg_log_lik(params: np.ndarray) -> float:
        beta = params[:n_params]
        log_phi = params[-1]
        # Bound log_phi to avoid overflow
        log_phi = float(np.clip(log_phi, -10.0, 10.0))
        phi = float(np.exp(log_phi))
        eta = X @ beta + offset
        eta = np.clip(eta, -50.0, 50.0)
        mu = np.exp(eta)
        mu = np.clip(mu, 1e-12, 1e12)

        # NB2 log-pmf:
        #   log Gamma(y+phi) - log Gamma(phi) - log Gamma(y+1)
        #     + phi*log(phi/(phi+mu)) + y*log(mu/(phi+mu))
        from scipy.special import gammaln
        ll = (
            gammaln(y_count + phi)
            - gammaln(phi)
            - gammaln(y_count + 1.0)
            + phi * (np.log(phi) - np.log(phi + mu))
            + y_count * (np.log(mu) - np.log(phi + mu))
        )
        nll = -float(np.sum(ll))

        # Ridge penalty on area dummies (random-intercept approximation)
        if n_area_re > 0:
            area_block = beta[idx_area_start:idx_area_end]
            ss = float(np.sum(area_block ** 2))
            tau2 = (ss + 2.0) / (n_area_re + 6.0)
            nll += 0.5 * ss / max(tau2, 1e-6)

        if not np.isfinite(nll):
            return 1e12
        return nll

    res = optimize.minimize(
        neg_log_lik,
        init,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": tol},
    )

    params_hat = res.x
    log_lik = -float(res.fun)
    converged = bool(res.success)

    # --- 7. Approximate SEs via numerical Hessian on the focal block ---
    # Full-Hessian SEs over thousands of area dummies are expensive and
    # not the point — we want SEs on (intercept, alpha_v, year FE).
    focal_idx = [idx_intercept, idx_alpha_v] + list(
        range(idx_year_start, idx_year_end)
    )

    def neg_log_lik_focal(focal_params: np.ndarray) -> float:
        p = params_hat.copy()
        for k, ix in enumerate(focal_idx):
            p[ix] = focal_params[k]
        return neg_log_lik(p)

    focal_hat = params_hat[focal_idx]
    H = _numerical_hessian(neg_log_lik_focal, focal_hat)
    try:
        cov = np.linalg.inv(H)
        focal_se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        focal_se = np.full(len(focal_idx), float("nan"))

    se_lookup = dict(zip(focal_idx, focal_se))

    # --- 8. Area-RE SD from the fitted dummies -------------------------
    if n_area_re > 0:
        area_hat = params_hat[idx_area_start:idx_area_end]
        area_re_sd = float(np.std(area_hat, ddof=1)) if n_area_re > 1 else 0.0
    else:
        area_re_sd = 0.0

    # --- 9. Pack coefficients ------------------------------------------
    alpha_v_coef = SMIForceDisparityCoefficient(
        name="alpha_v",
        estimate=float(params_hat[idx_alpha_v]),
        std_error=float(se_lookup.get(idx_alpha_v, float("nan"))),
    )
    intercept_coef = SMIForceDisparityCoefficient(
        name="mu",
        estimate=float(params_hat[idx_intercept]),
        std_error=float(se_lookup.get(idx_intercept, float("nan"))),
    )
    year_effects = []
    if include_year_fe:
        for k, y in enumerate(non_baseline_years):
            ix = idx_year_start + k
            year_effects.append(SMIForceDisparityCoefficient(
                name=f"delta_{y}",
                estimate=float(params_hat[ix]),
                std_error=float(se_lookup.get(ix, float("nan"))),
            ))

    dispersion = float(np.exp(np.clip(params_hat[-1], -10.0, 10.0)))

    exposure_summary = {
        "smi_exposure_mean": float(smi_exposure.mean()),
        "smi_exposure_median": float(smi_exposure.median()),
        "smi_exposure_min": float(smi_exposure.min()),
        "smi_exposure_max": float(smi_exposure.max()),
        "n_areas": int(area_frame.shape[0]),
        "baseline_year": int(baseline_year),
        "baseline_area": baseline_area,
    }
    if return_design:
        exposure_summary["X"] = X
        exposure_summary["y"] = y_count
        exposure_summary["offset"] = offset
        exposure_summary["design_cols"] = design_cols

    note = (
        "Frequentist MLE approximation to the paper's Bayesian "
        "hierarchical NB.  Area random intercept implemented as "
        "ridge-penalised dummies; SDs reported are empirical (not "
        "posterior).  Pass return_design=True to hand off to pymc/brms "
        "for paper-grade credible intervals."
    )

    return SMIForceDisparityResult(
        alpha_v=alpha_v_coef,
        intercept=intercept_coef,
        year_effects=year_effects,
        area_random_effect_sd=area_re_sd,
        dispersion=dispersion,
        n_events=int(y_count.sum()),
        n_area_years=int(n_obs),
        log_likelihood=log_lik,
        converged=converged,
        exposure_summary=exposure_summary,
        note=note,
    )


def _numerical_hessian(
    fn,
    x: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """Central-difference Hessian for SE extraction on the focal block."""
    n = x.size
    H = np.zeros((n, n))
    f0 = fn(x)
    for i in range(n):
        for j in range(i, n):
            x_pp = x.copy(); x_pp[i] += eps; x_pp[j] += eps
            x_pm = x.copy(); x_pm[i] += eps; x_pm[j] -= eps
            x_mp = x.copy(); x_mp[i] -= eps; x_mp[j] += eps
            x_mm = x.copy(); x_mm[i] -= eps; x_mm[j] -= eps
            val = (fn(x_pp) - fn(x_pm) - fn(x_mp) + fn(x_mm)) / (4 * eps * eps)
            H[i, j] = val
            H[j, i] = val
    # Tiny Tikhonov regulariser for numerical stability
    H = H + 1e-8 * np.eye(n)
    _ = f0  # silence linter
    return H
