"""Sensitivity analysis for causal inference assumptions.

Provides tools to assess the robustness of causal effect estimates to
unmeasured confounding, model specification, and other threats to
internal validity.  Implements Rosenbaum bounds, E-value, Ding-VanderWeele
bias formulas, tipping-point analysis, and specification curve analysis.

References
----------
* Rosenbaum (2002). *Observational Studies*, 2nd ed.
* VanderWeele & Ding (2017). Sensitivity analysis in observational research.
* Cinelli & Hazlett (2020). Making sense of sensitivity.
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
class EValueResult:
    """Result from E-value computation."""

    point_estimate: float
    e_value_point: float
    e_value_ci: float
    rr: float
    ci_lower: float
    ci_upper: float
    interpretation: str


@dataclass
class RosenbaumBounds:
    """Result from Rosenbaum sensitivity analysis."""

    gamma_values: np.ndarray
    p_upper: np.ndarray
    p_lower: np.ndarray
    critical_gamma: float
    method: str
    interpretation: str


@dataclass
class TippingPointResult:
    """Result from tipping-point analysis."""

    delta_values: np.ndarray
    adjusted_estimates: np.ndarray
    adjusted_p_values: np.ndarray
    tipping_point: float
    original_estimate: float
    interpretation: str


@dataclass
class OmittedVariableBias:
    """Result from omitted variable bias analysis (Cinelli & Hazlett)."""

    estimate: float
    se: float
    rv_q: float  # robustness value at q
    rv_qa: float  # robustness value at q,alpha
    partial_r2_treatment: float
    benchmark_bounds: dict[str, tuple[float, float]]
    interpretation: str


@dataclass
class SpecificationCurveResult:
    """Result from specification curve analysis."""

    estimates: np.ndarray
    ses: np.ndarray
    p_values: np.ndarray
    specifications: list[dict]
    median_estimate: float
    iqr_lower: float
    iqr_upper: float
    pct_significant: float
    pct_same_sign: float


# ---------------------------------------------------------------------------
# E-value (VanderWeele & Ding 2017)
# ---------------------------------------------------------------------------


def _rr_to_evalue(rr: float) -> float:
    """Convert a risk ratio to its E-value."""
    if rr < 1:
        rr = 1 / rr
    return rr + np.sqrt(rr * (rr - 1))


def e_value_rr(
    rr: float,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
) -> EValueResult:
    """Compute E-value for a risk ratio.

    The E-value is the minimum strength of association (on the risk ratio
    scale) that an unmeasured confounder would need to have with both
    treatment and outcome to fully explain away the observed association.

    Parameters
    ----------
    rr : float
        Observed risk ratio.
    ci_lower : float, optional
        Lower bound of 95% CI for the RR.
    ci_upper : float, optional
        Upper bound of 95% CI for the RR.

    Returns
    -------
    EValueResult
    """
    e_point = _rr_to_evalue(rr)

    if ci_lower is not None and ci_upper is not None:
        # E-value for the CI limit closest to 1.
        if rr >= 1:
            e_ci = _rr_to_evalue(ci_lower) if ci_lower > 1 else 1.0
        else:
            e_ci = _rr_to_evalue(ci_upper) if ci_upper < 1 else 1.0
    else:
        e_ci = float("nan")
        ci_lower = ci_lower or float("nan")
        ci_upper = ci_upper or float("nan")

    interpretation = (
        f"An unmeasured confounder would need RR >= {e_point:.2f} with both "
        f"treatment and outcome to explain away the point estimate (RR={rr:.2f}). "
        f"To move the CI to include the null, RR >= {e_ci:.2f} would be needed."
    )

    return EValueResult(
        point_estimate=rr,
        e_value_point=e_point,
        e_value_ci=e_ci,
        rr=rr,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        interpretation=interpretation,
    )


def e_value_or(
    odds_ratio: float,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
    prevalence: float | None = None,
) -> EValueResult:
    """Compute E-value for an odds ratio.

    For rare outcomes (prevalence < 15%), the OR approximates the RR.
    For common outcomes, a correction is applied using the prevalence.

    Parameters
    ----------
    odds_ratio : float
        Observed odds ratio.
    ci_lower : float, optional
        Lower CI bound.
    ci_upper : float, optional
        Upper CI bound.
    prevalence : float, optional
        Outcome prevalence for OR-to-RR conversion.

    Returns
    -------
    EValueResult
    """
    if prevalence is not None and prevalence >= 0.15:
        # Convert OR to RR using Zhang & Yu (1998) formula.
        rr = odds_ratio / (1 - prevalence + prevalence * odds_ratio)
        if ci_lower is not None:
            ci_lower = ci_lower / (1 - prevalence + prevalence * ci_lower)
        if ci_upper is not None:
            ci_upper = ci_upper / (1 - prevalence + prevalence * ci_upper)
    else:
        rr = odds_ratio
        # OR ≈ RR for rare outcomes.

    return e_value_rr(rr, ci_lower, ci_upper)


def e_value_hr(
    hr: float,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
) -> EValueResult:
    """Compute E-value for a hazard ratio.

    Uses the HR-to-RR approximation from VanderWeele (2017).

    Parameters
    ----------
    hr : float
        Observed hazard ratio.
    ci_lower : float, optional
        Lower CI bound.
    ci_upper : float, optional
        Upper CI bound.

    Returns
    -------
    EValueResult
    """
    # HR-to-RR approximation.
    rr = (1 - 0.5 ** np.sqrt(hr)) / (1 - 0.5 ** np.sqrt(1 / hr)) if hr != 1 else 1.0
    if ci_lower is not None and ci_lower > 0:
        rr_lo = (1 - 0.5 ** np.sqrt(ci_lower)) / (1 - 0.5 ** np.sqrt(1 / ci_lower)) if ci_lower != 1 else 1.0
    else:
        rr_lo = None
    if ci_upper is not None and ci_upper > 0:
        rr_hi = (1 - 0.5 ** np.sqrt(ci_upper)) / (1 - 0.5 ** np.sqrt(1 / ci_upper)) if ci_upper != 1 else 1.0
    else:
        rr_hi = None
    return e_value_rr(rr, rr_lo, rr_hi)


def e_value_d(
    d: float,
    se: float | None = None,
    n: int | None = None,
) -> EValueResult:
    """Compute E-value for a standardized mean difference (Cohen's d).

    Converts d to a risk ratio scale using the VanderWeele & Ding
    approximation: RR ≈ exp(0.91 * d).

    Parameters
    ----------
    d : float
        Standardized mean difference.
    se : float, optional
        Standard error of d.
    n : int, optional
        Sample size (used to compute SE if not provided).

    Returns
    -------
    EValueResult
    """
    rr = np.exp(0.91 * d)

    if se is not None:
        rr_lo = np.exp(0.91 * (d - 1.96 * se))
        rr_hi = np.exp(0.91 * (d + 1.96 * se))
    elif n is not None:
        se_approx = np.sqrt(4 / n)
        rr_lo = np.exp(0.91 * (d - 1.96 * se_approx))
        rr_hi = np.exp(0.91 * (d + 1.96 * se_approx))
    else:
        rr_lo = None
        rr_hi = None

    return e_value_rr(rr, rr_lo, rr_hi)


# ---------------------------------------------------------------------------
# Rosenbaum bounds
# ---------------------------------------------------------------------------


def rosenbaum_bounds(
    treated_outcomes: np.ndarray,
    control_outcomes: np.ndarray,
    gamma_range: np.ndarray | None = None,
    method: str = "wilcoxon",
) -> RosenbaumBounds:
    """Rosenbaum sensitivity analysis for matched pair designs.

    Tests how strong hidden bias (Gamma) would need to be to alter
    the study's conclusions.

    Parameters
    ----------
    treated_outcomes : array-like
        Outcomes for treated units in matched pairs.
    control_outcomes : array-like
        Outcomes for control units in matched pairs.
    gamma_range : array-like, optional
        Range of Gamma values to test.  Default: 1.0 to 5.0 by 0.25.
    method : str
        Test method: 'wilcoxon' (default), 'sign', 'mcnemar'.

    Returns
    -------
    RosenbaumBounds
    """
    t = np.asarray(treated_outcomes, dtype=float)
    c = np.asarray(control_outcomes, dtype=float)
    n = len(t)
    diffs = t - c

    if gamma_range is None:
        gamma_range = np.arange(1.0, 5.25, 0.25)

    gamma_range = np.asarray(gamma_range, dtype=float)
    p_upper = np.empty(len(gamma_range))
    p_lower = np.empty(len(gamma_range))

    if method == "wilcoxon":
        ranks = stats.rankdata(np.abs(diffs))
        signs = np.sign(diffs)
        t_obs = np.sum(ranks[signs > 0])

        for i, gamma in enumerate(gamma_range):
            # Under Gamma-bias, the probability that each pair is treated
            # is between 1/(1+Gamma) and Gamma/(1+Gamma).
            p_treat = gamma / (1 + gamma)

            # Upper bound: maximize p-value.
            expected_upper = np.sum(ranks * p_treat)
            var_upper = np.sum(ranks**2 * p_treat * (1 - p_treat))
            z_upper = (t_obs - expected_upper) / np.sqrt(max(var_upper, 1e-10))
            p_upper[i] = 1 - stats.norm.cdf(z_upper)

            # Lower bound.
            p_treat_low = 1 / (1 + gamma)
            expected_lower = np.sum(ranks * p_treat_low)
            var_lower = np.sum(ranks**2 * p_treat_low * (1 - p_treat_low))
            z_lower = (t_obs - expected_lower) / np.sqrt(max(var_lower, 1e-10))
            p_lower[i] = 1 - stats.norm.cdf(z_lower)

    elif method == "sign":
        n_positive = np.sum(diffs > 0)

        for i, gamma in enumerate(gamma_range):
            p_treat = gamma / (1 + gamma)
            # Upper bound: binomial test.
            p_upper[i] = 1 - stats.binom.cdf(n_positive - 1, n, p_treat)
            p_treat_low = 1 / (1 + gamma)
            p_lower[i] = 1 - stats.binom.cdf(n_positive - 1, n, p_treat_low)

    elif method == "mcnemar":
        # For binary outcomes: discordant pairs.
        b = np.sum((t == 1) & (c == 0))  # treated=1, control=0
        cc = np.sum((t == 0) & (c == 1))  # treated=0, control=1
        n_disc = b + cc

        for i, gamma in enumerate(gamma_range):
            p_treat = gamma / (1 + gamma)
            p_upper[i] = 1 - stats.binom.cdf(int(b) - 1, int(n_disc), p_treat)
            p_treat_low = 1 / (1 + gamma)
            p_lower[i] = 1 - stats.binom.cdf(int(b) - 1, int(n_disc), p_treat_low)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Critical Gamma: smallest Gamma where upper bound p > 0.05.
    critical_idx = np.where(p_upper > 0.05)[0]
    critical_gamma = float(gamma_range[critical_idx[0]]) if len(critical_idx) > 0 else float(gamma_range[-1])

    interpretation = (
        f"The study conclusion is sensitive to hidden bias at Gamma = {critical_gamma:.2f}. "
        f"An unobserved covariate that changes the odds of treatment by a factor of "
        f"{critical_gamma:.2f} could explain away the result."
    )

    return RosenbaumBounds(
        gamma_values=gamma_range,
        p_upper=p_upper,
        p_lower=p_lower,
        critical_gamma=critical_gamma,
        method=method,
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# Tipping point analysis
# ---------------------------------------------------------------------------


def tipping_point_analysis(
    estimate: float,
    se: float,
    n_treated: int,
    n_control: int,
    delta_range: np.ndarray | None = None,
    outcome_type: str = "continuous",
) -> TippingPointResult:
    """Tipping-point analysis for missing data sensitivity.

    Evaluates how much the treatment effect changes if missing outcomes
    are systematically different from observed outcomes.

    Parameters
    ----------
    estimate : float
        Observed treatment effect estimate.
    se : float
        Standard error of the estimate.
    n_treated : int
        Number of treated units.
    n_control : int
        Number of control units.
    delta_range : array-like, optional
        Range of bias parameters to evaluate.
    outcome_type : str
        'continuous' or 'binary'.

    Returns
    -------
    TippingPointResult
    """
    if delta_range is None:
        max_delta = abs(estimate) * 3
        delta_range = np.linspace(-max_delta, max_delta, 101)

    delta_range = np.asarray(delta_range, dtype=float)
    n_total = n_treated + n_control

    adjusted_estimates = estimate - delta_range
    adjusted_z = adjusted_estimates / se
    adjusted_p = 2 * (1 - stats.norm.cdf(np.abs(adjusted_z)))

    # Tipping point: delta where adjusted p-value crosses 0.05.
    significant = adjusted_p <= 0.05
    if significant.all():
        tipping_point = float(delta_range[-1])
    elif not significant.any():
        tipping_point = float(delta_range[0])
    else:
        # Find transition.
        transitions = np.diff(significant.astype(int))
        cross_idx = np.where(transitions != 0)[0]
        if len(cross_idx) > 0:
            tipping_point = float(delta_range[cross_idx[0]])
        else:
            tipping_point = float("nan")

    _robust = abs(tipping_point) > abs(estimate)
    _robustness_msg = (
        "This suggests the result is robust."
        if _robust
        else "This suggests the result may be sensitive to missing data."
    )
    interpretation = (
        f"The observed estimate ({estimate:.4f}) becomes non-significant "
        f"when outcomes for missing data differ by delta = {tipping_point:.4f}. "
        f"{_robustness_msg}"
    )

    return TippingPointResult(
        delta_values=delta_range,
        adjusted_estimates=adjusted_estimates,
        adjusted_p_values=adjusted_p,
        tipping_point=tipping_point,
        original_estimate=estimate,
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# Omitted variable bias (Cinelli & Hazlett 2020)
# ---------------------------------------------------------------------------


def omitted_variable_bias(
    estimate: float,
    se: float,
    dof: int,
    r2_yd_x: float,
    partial_r2_treatment: float,
    q: float = 1.0,
    alpha: float = 0.05,
    benchmark_covariates: dict[str, float] | None = None,
) -> OmittedVariableBias:
    """Omitted variable bias analysis (sensemakr framework).

    Implements the Cinelli & Hazlett (2020) approach to assess how much
    an unobserved confounder would need to explain to change the conclusion.

    Parameters
    ----------
    estimate : float
        Treatment coefficient estimate.
    se : float
        Standard error of the estimate.
    dof : int
        Residual degrees of freedom.
    r2_yd_x : float
        Partial R-squared of treatment with outcome (controlling for X).
    partial_r2_treatment : float
        Same as r2_yd_x (for clarity).
    q : float
        Fraction of the estimate to be explained away.  Default 1.0 (full).
    alpha : float
        Significance level.
    benchmark_covariates : dict, optional
        Dictionary mapping covariate names to their partial R-squared
        with the outcome.  Used to generate benchmark bounds.

    Returns
    -------
    OmittedVariableBias
    """
    t_stat = estimate / se
    f_stat = t_stat**2

    # Robustness value (RV_q): partial R2 of confounder needed to
    # reduce estimate by proportion q.
    rv_q = 0.5 * (np.sqrt(f_stat**2 - f_stat) - f_stat + 1) if f_stat > 1 else 0.0
    rv_q = max(rv_q, 0.0)

    # RV_{q,alpha}: partial R2 needed to make CI include q*estimate.
    t_crit = stats.t.ppf(1 - alpha / 2, dof)
    f_crit = t_crit**2
    rv_qa = 0.5 * (np.sqrt(f_stat**2 - f_crit * f_stat) - f_stat + f_crit) if f_stat > f_crit else 0.0
    rv_qa = max(rv_qa, 0.0)

    # Benchmark bounds.
    bounds = {}
    if benchmark_covariates:
        for name, r2_bench in benchmark_covariates.items():
            # If confounder is as strong as benchmark covariate:
            bias = estimate * r2_bench / partial_r2_treatment if partial_r2_treatment > 0 else 0
            bounds[name] = (
                estimate - bias,
                estimate + bias,
            )

    interpretation = (
        f"To explain away {q * 100:.0f}% of the estimate ({estimate:.4f}), "
        f"an unobserved confounder would need partial R2 >= {rv_q:.4f} with both "
        f"treatment and outcome. To make the CI include zero, partial R2 >= {rv_qa:.4f}."
    )

    return OmittedVariableBias(
        estimate=estimate,
        se=se,
        rv_q=rv_q,
        rv_qa=rv_qa,
        partial_r2_treatment=partial_r2_treatment,
        benchmark_bounds=bounds,
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# Specification curve analysis
# ---------------------------------------------------------------------------


def specification_curve(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    covariate_sets: list[list[str]],
    sample_filters: list[tuple[str, callable]] | None = None,
    model_types: list[str] | None = None,
    alpha: float = 0.05,
) -> SpecificationCurveResult:
    """Run a specification curve analysis.

    Estimates the treatment effect across many reasonable model specifications
    to assess robustness.

    Parameters
    ----------
    data : pd.DataFrame
        Analysis dataset.
    outcome : str
        Outcome variable name.
    treatment : str
        Treatment variable name.
    covariate_sets : list of list[str]
        Different sets of covariates to try.
    sample_filters : list of (name, callable), optional
        Different sample restrictions to try.
    model_types : list[str], optional
        Model types: 'ols', 'logistic', 'robust'.  Default: ['ols'].
    alpha : float
        Significance level.

    Returns
    -------
    SpecificationCurveResult
    """
    import statsmodels.api as sm

    if model_types is None:
        model_types = ["ols"]
    if sample_filters is None:
        sample_filters = [("full_sample", lambda df: df)]

    estimates = []
    ses = []
    p_values = []
    specifications = []

    for filter_name, filter_fn in sample_filters:
        filtered = filter_fn(data).copy()
        if len(filtered) < 10:
            continue

        for cov_set in covariate_sets:
            # Check all covariates exist.
            missing_cols = [c for c in cov_set if c not in filtered.columns]
            if missing_cols:
                continue

            for model_type in model_types:
                try:
                    y = filtered[outcome].astype(float)
                    X_vars = [treatment] + cov_set
                    X = filtered[X_vars].astype(float)
                    X = sm.add_constant(X, has_constant="add")

                    # Drop any rows with NaN.
                    mask = y.notna() & X.notna().all(axis=1)
                    y = y[mask]
                    X = X.loc[mask]

                    if len(y) < len(X_vars) + 2:
                        continue

                    if model_type == "ols":
                        model = sm.OLS(y, X).fit()
                    elif model_type == "logistic":
                        model = sm.Logit(y, X).fit(disp=0)
                    elif model_type == "robust":
                        model = sm.RLM(y, X).fit()
                    else:
                        continue

                    est = model.params[treatment]
                    se = model.bse[treatment]
                    pval = model.pvalues[treatment]

                    estimates.append(est)
                    ses.append(se)
                    p_values.append(pval)
                    specifications.append(
                        {
                            "sample": filter_name,
                            "covariates": cov_set,
                            "model": model_type,
                            "n": int(len(y)),
                            "estimate": est,
                            "se": se,
                            "p_value": pval,
                        }
                    )

                except Exception:
                    continue

    if not estimates:
        return SpecificationCurveResult(
            estimates=np.array([]),
            ses=np.array([]),
            p_values=np.array([]),
            specifications=[],
            median_estimate=float("nan"),
            iqr_lower=float("nan"),
            iqr_upper=float("nan"),
            pct_significant=0.0,
            pct_same_sign=0.0,
        )

    est_arr = np.array(estimates)
    se_arr = np.array(ses)
    p_arr = np.array(p_values)

    median_est = float(np.median(est_arr))
    q25, q75 = np.percentile(est_arr, [25, 75])

    n_sig = np.sum(p_arr <= alpha)
    modal_sign = np.sign(median_est)
    n_same_sign = np.sum(np.sign(est_arr) == modal_sign)

    return SpecificationCurveResult(
        estimates=est_arr,
        ses=se_arr,
        p_values=p_arr,
        specifications=specifications,
        median_estimate=median_est,
        iqr_lower=float(q25),
        iqr_upper=float(q75),
        pct_significant=float(n_sig / len(est_arr) * 100),
        pct_same_sign=float(n_same_sign / len(est_arr) * 100),
    )


# ---------------------------------------------------------------------------
# Manski bounds (partial identification)
# ---------------------------------------------------------------------------


def manski_bounds(
    outcome_treated: np.ndarray,
    outcome_control: np.ndarray,
    p_treated: float,
    outcome_range: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Compute Manski worst-case bounds for the ATE.

    Under no assumptions about selection, the ATE is only partially
    identified.

    Parameters
    ----------
    outcome_treated : array-like
        Outcomes for treated units.
    outcome_control : array-like
        Outcomes for control units.
    p_treated : float
        Proportion treated.
    outcome_range : tuple[float, float], optional
        Logical bounds on the outcome.  Default: (0, 1) for binary.

    Returns
    -------
    dict
        With keys: lower_bound, upper_bound, point_estimate, width.
    """
    y1 = np.asarray(outcome_treated, dtype=float)
    y0 = np.asarray(outcome_control, dtype=float)

    if outcome_range is None:
        outcome_range = (0.0, 1.0)

    y_min, y_max = outcome_range
    e1 = np.mean(y1)
    e0 = np.mean(y0)

    # Manski bounds: ATE ∈ [E[Y|T=1]*P(T=1) + y_min*P(T=0) - E[Y|T=0]*P(T=0) - y_max*P(T=1),
    #                        E[Y|T=1]*P(T=1) + y_max*P(T=0) - E[Y|T=0]*P(T=0) - y_min*P(T=1)]
    p1 = p_treated
    p0 = 1 - p_treated

    lower = e1 * p1 + y_min * p0 - (e0 * p0 + y_max * p1)
    upper = e1 * p1 + y_max * p0 - (e0 * p0 + y_min * p1)

    # Simplification for the standard case:
    lower_simple = e1 - e0 - (y_max - y_min) * (1 - p1)
    upper_simple = e1 - e0 + (y_max - y_min) * p1

    return {
        "lower_bound": float(min(lower, lower_simple)),
        "upper_bound": float(max(upper, upper_simple)),
        "point_estimate": float(e1 - e0),
        "width": float(max(upper, upper_simple) - min(lower, lower_simple)),
    }


# ---------------------------------------------------------------------------
# Bias-adjusted estimates
# ---------------------------------------------------------------------------


def bias_adjusted_estimate(
    estimate: float,
    se: float,
    rr_ud: float,
    rr_eu: float,
    prevalence_confounder: float = 0.5,
) -> dict[str, float]:
    """Compute bias-adjusted treatment effect.

    Uses the Ding & VanderWeele (2016) bias formula.

    Parameters
    ----------
    estimate : float
        Observed treatment effect (on log-RR or coefficient scale).
    se : float
        Standard error.
    rr_ud : float
        Risk ratio relating confounder to outcome.
    rr_eu : float
        Risk ratio relating treatment to confounder.
    prevalence_confounder : float
        Prevalence of the confounder in the population.

    Returns
    -------
    dict
        With keys: adjusted_estimate, bias, adjusted_ci_lower, adjusted_ci_upper.
    """
    # Bias formula: B ≈ (RR_UD * RR_EU - 1) / (RR_UD + RR_EU - 1) * prevalence adjustment
    bias_factor = (rr_ud * rr_eu - 1) / max(rr_ud + rr_eu - 1, 0.01)
    bias = np.log(bias_factor) * prevalence_confounder

    adjusted = estimate - bias
    ci_lo = adjusted - 1.96 * se
    ci_hi = adjusted + 1.96 * se

    return {
        "adjusted_estimate": float(adjusted),
        "bias": float(bias),
        "adjusted_ci_lower": float(ci_lo),
        "adjusted_ci_upper": float(ci_hi),
        "original_estimate": float(estimate),
    }


# ---------------------------------------------------------------------------
# Quantitative bias analysis (probabilistic)
# ---------------------------------------------------------------------------


def probabilistic_bias_analysis(
    estimate: float,
    se: float,
    n_simulations: int = 10000,
    bias_parms: dict[str, tuple[float, float]] | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """Probabilistic (Monte Carlo) sensitivity analysis.

    Draws bias parameters from specified prior distributions and computes
    the distribution of bias-adjusted estimates.

    Parameters
    ----------
    estimate : float
        Observed estimate.
    se : float
        Standard error.
    n_simulations : int
        Number of Monte Carlo draws.
    bias_parms : dict, optional
        Parameters as (mean, sd) tuples for: 'rr_ud', 'rr_eu', 'prevalence'.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Summary of bias-adjusted estimate distribution.
    """
    rng = np.random.default_rng(seed)

    if bias_parms is None:
        bias_parms = {
            "rr_ud": (1.5, 0.3),
            "rr_eu": (1.5, 0.3),
            "prevalence": (0.3, 0.1),
        }

    rr_ud = np.abs(rng.normal(bias_parms["rr_ud"][0], bias_parms["rr_ud"][1], n_simulations))
    rr_eu = np.abs(rng.normal(bias_parms["rr_eu"][0], bias_parms["rr_eu"][1], n_simulations))
    prev = np.clip(rng.normal(bias_parms["prevalence"][0], bias_parms["prevalence"][1], n_simulations), 0.01, 0.99)

    # Add random error.
    estimates_with_error = rng.normal(estimate, se, n_simulations)

    # Compute bias-adjusted estimates.
    bias_factors = (rr_ud * rr_eu - 1) / np.maximum(rr_ud + rr_eu - 1, 0.01)
    biases = np.log(np.maximum(bias_factors, 0.01)) * prev
    adjusted = estimates_with_error - biases

    return {
        "original_estimate": float(estimate),
        "median_adjusted": float(np.median(adjusted)),
        "mean_adjusted": float(np.mean(adjusted)),
        "ci_2.5": float(np.percentile(adjusted, 2.5)),
        "ci_97.5": float(np.percentile(adjusted, 97.5)),
        "pct_null_included": float(np.mean((adjusted < 0) != (estimate < 0)) * 100),
        "pct_same_sign": float(np.mean(np.sign(adjusted) == np.sign(estimate)) * 100),
        "n_simulations": n_simulations,
    }


# ---------------------------------------------------------------------------
# Cross-validation of sensitivity
# ---------------------------------------------------------------------------


def sensitivity_summary(
    estimate: float,
    se: float,
    rr: float | None = None,
    odds_ratio: float | None = None,
    hazard_ratio: float | None = None,
    prevalence: float | None = None,
) -> pd.DataFrame:
    """Generate a comprehensive sensitivity analysis summary.

    Computes E-values, tipping points, and robustness metrics for
    a single treatment effect estimate.

    Parameters
    ----------
    estimate : float
        Treatment effect estimate.
    se : float
        Standard error.
    rr : float, optional
        Risk ratio (if applicable).
    odds_ratio : float, optional
        Odds ratio (if applicable).
    hazard_ratio : float, optional
        Hazard ratio (if applicable).
    prevalence : float, optional
        Outcome prevalence.

    Returns
    -------
    pd.DataFrame
        Summary table of sensitivity metrics.
    """
    rows = []

    # Basic info.
    ci_lo = estimate - 1.96 * se
    ci_hi = estimate + 1.96 * se
    z = estimate / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    rows.append({"metric": "estimate", "value": estimate})
    rows.append({"metric": "se", "value": se})
    rows.append({"metric": "ci_lower", "value": ci_lo})
    rows.append({"metric": "ci_upper", "value": ci_hi})
    rows.append({"metric": "p_value", "value": p})

    # E-values.
    if rr is not None:
        ev = e_value_rr(rr, ci_lo if rr >= 1 else None, ci_hi if rr >= 1 else None)
        rows.append({"metric": "e_value_point", "value": ev.e_value_point})
        rows.append({"metric": "e_value_ci", "value": ev.e_value_ci})

    if odds_ratio is not None:
        ev = e_value_or(odds_ratio, prevalence=prevalence)
        rows.append({"metric": "e_value_or_point", "value": ev.e_value_point})
        rows.append({"metric": "e_value_or_ci", "value": ev.e_value_ci})

    if hazard_ratio is not None:
        ev = e_value_hr(hazard_ratio)
        rows.append({"metric": "e_value_hr_point", "value": ev.e_value_point})
        rows.append({"metric": "e_value_hr_ci", "value": ev.e_value_ci})

    # Tipping point.
    tp = tipping_point_analysis(estimate, se, 100, 100)
    rows.append({"metric": "tipping_point_delta", "value": tp.tipping_point})

    return pd.DataFrame(rows)
