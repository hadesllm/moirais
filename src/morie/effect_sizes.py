"""
Comprehensive effect-size calculations for epidemiological research.

This module provides every major family of effect-size estimators used in
biomedical and social-science research, each with analytic or bootstrap
confidence intervals.  Functions are organised by the type of comparison:

- **Standardised mean differences**: Cohen's *d*, Hedges' *g*, Glass's delta
- **Common-language effect sizes**: CLES / probability of superiority
- **Correlation-based**: *r*, *R*-squared, eta-squared, partial eta-squared,
  omega-squared, epsilon-squared
- **Contingency-table measures**: odds ratio, risk ratio, risk difference, NNT,
  NNH, rate ratio, incidence rate difference
- **Association measures**: Cohen's *w*, Cramer's *V*, phi coefficient
- **Non-parametric**: rank-biserial correlation, Cliff's delta, Vargha--Delaney *A*
- **Regression**: standardised coefficients, coefficient of variation
- **Conversion**: *d* <-> *r*, OR <-> *d*, etc.
- **Meta-analysis**: fixed- and random-effects pooling, *I*-squared,
  prediction intervals

References
----------
Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*
    (2nd ed.). Lawrence Erlbaum Associates.
Hedges, L. V., & Olkin, I. (1985). *Statistical Methods for Meta-Analysis*.
    Academic Press.
Borenstein, M., Hedges, L. V., Higgins, J. P. T., & Rothstein, H. R. (2009).
    *Introduction to Meta-Analysis*. Wiley.
Vargha, A., & Delaney, H. D. (2000). A critique and improvement of the CL
    common language effect size statistics of McGraw and Wong. *JEBS*, 25(2),
    101--132.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import pandas as pd
import scipy.stats as stats

logger = logging.getLogger(__name__)


# ===================================================================
# Result container
# ===================================================================


@dataclass
class EffectSizeResult:
    """Standardised result for every effect-size calculation.

    Parameters
    ----------
    measure : str
        Name of the effect-size statistic.
    estimate : float
        Point estimate.
    ci_lower : float | None
        Lower confidence bound.
    ci_upper : float | None
        Upper confidence bound.
    se : float | None
        Standard error (analytic or bootstrap).
    n : int | None
        Sample size used.
    extra : dict
        Additional outputs.
    """

    measure: str
    estimate: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    se: float | None = None
    n: int | None = None
    extra: dict = field(default_factory=dict)


# ===================================================================
# Helpers
# ===================================================================


def _arr(x: Union[np.ndarray, pd.Series, list]) -> np.ndarray:
    """Coerce to float64 array, drop NaN."""
    a = np.asarray(x, dtype=np.float64).ravel()
    return a[np.isfinite(a)]


def _bootstrap_ci(
    func,
    args: tuple,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI and SE for an effect-size function.

    Parameters
    ----------
    func : callable
        Function that takes resampled arrays and returns a scalar.
    args : tuple of arrays
        Original arrays to bootstrap.
    n_boot : int
    confidence : float
    seed : int

    Returns
    -------
    tuple
        (se, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    boot_vals = np.empty(n_boot)
    arrays = [np.asarray(a) for a in args]
    for b in range(n_boot):
        resampled = tuple(a[rng.choice(len(a), len(a), replace=True)] for a in arrays)
        try:
            boot_vals[b] = func(*resampled)
        except Exception:
            boot_vals[b] = np.nan
    boot_vals = boot_vals[np.isfinite(boot_vals)]
    if len(boot_vals) == 0:
        return 0.0, np.nan, np.nan
    alpha = (1 - confidence) / 2
    ci_lo = float(np.percentile(boot_vals, 100 * alpha))
    ci_hi = float(np.percentile(boot_vals, 100 * (1 - alpha)))
    se = float(np.std(boot_vals, ddof=1))
    return se, ci_lo, ci_hi


# ===================================================================
# STANDARDISED MEAN DIFFERENCES
# ===================================================================


def cohens_d(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Cohen's *d* for independent samples (pooled SD denominator).

    .. math::
        d = \\frac{\\bar{x} - \\bar{y}}{s_p}

    Parameters
    ----------
    x, y : array-like
        Two independent samples.
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    x, y = _arr(x), _arr(y)
    nx, ny = len(x), len(y)
    sp = math.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    d = (x.mean() - y.mean()) / sp if sp > 0 else 0.0
    # Analytic SE (Hedges & Olkin, 1985)
    se = math.sqrt((nx + ny) / (nx * ny) + d**2 / (2 * (nx + ny - 2)))
    z = stats.norm.ppf((1 + confidence) / 2)
    return EffectSizeResult(
        measure="Cohen's d",
        estimate=float(d),
        ci_lower=float(d - z * se),
        ci_upper=float(d + z * se),
        se=float(se),
        n=nx + ny,
    )


def hedges_g(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Hedges' *g* -- bias-corrected Cohen's *d*.

    Applies the exact correction factor :math:`J = 1 - 3/(4(n_1+n_2-2)-1)`.

    Parameters
    ----------
    x, y : array-like
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    x, y = _arr(x), _arr(y)
    d_result = cohens_d(x, y, confidence)
    df_val = len(x) + len(y) - 2
    J = 1 - 3 / (4 * df_val - 1) if df_val > 1 else 1.0
    g = d_result.estimate * J
    se = d_result.se * J if d_result.se else 0.0
    z = stats.norm.ppf((1 + confidence) / 2)
    return EffectSizeResult(
        measure="Hedges' g",
        estimate=float(g),
        ci_lower=float(g - z * se),
        ci_upper=float(g + z * se),
        se=float(se),
        n=d_result.n,
        extra={"correction_factor": float(J)},
    )


def glass_delta(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    control: str = "y",
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Glass's delta -- uses the control group SD as denominator.

    Parameters
    ----------
    x, y : array-like
    control : str, default "y"
        Which group is the control: ``"x"`` or ``"y"``.
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    x, y = _arr(x), _arr(y)
    ctrl = y if control == "y" else x
    sd_ctrl = ctrl.std(ddof=1)
    delta = (x.mean() - y.mean()) / sd_ctrl if sd_ctrl > 0 else 0.0
    n_ctrl = len(ctrl)
    se = math.sqrt(1 / len(x) + 1 / len(y) + delta**2 / (2 * (n_ctrl - 1)))
    z = stats.norm.ppf((1 + confidence) / 2)
    return EffectSizeResult(
        measure="Glass's delta",
        estimate=float(delta),
        ci_lower=float(delta - z * se),
        ci_upper=float(delta + z * se),
        se=float(se),
        n=len(x) + len(y),
    )


# ===================================================================
# COMMON LANGUAGE EFFECT SIZE
# ===================================================================


def cles(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Common Language Effect Size (probability of superiority).

    Estimates :math:`P(X > Y)` for randomly drawn observations from each group.

    Parameters
    ----------
    x, y : array-like
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    x, y = _arr(x), _arr(y)
    nx, ny = len(x), len(y)
    count = 0
    ties = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                count += 1
            elif xi == yj:
                ties += 1
    p_sup = (count + 0.5 * ties) / (nx * ny) if nx * ny > 0 else 0.5
    se, ci_lo, ci_hi = _bootstrap_ci(
        lambda a, b: sum(1 for ai in a for bj in b if ai > bj) / (len(a) * len(b)),
        (x, y),
        confidence=confidence,
    )
    return EffectSizeResult(
        measure="CLES (Prob. of superiority)",
        estimate=float(p_sup),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        se=se,
        n=nx + ny,
    )


# ===================================================================
# CORRELATION-BASED
# ===================================================================


def r_effect_size(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Pearson *r* as an effect size with Fisher *z* CI.

    Parameters
    ----------
    x, y : array-like
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    x, y = _arr(x), _arr(y)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    r, _ = stats.pearsonr(x, y)
    z_r = np.arctanh(r)
    se_z = 1 / math.sqrt(n - 3) if n > 3 else np.inf
    z_crit = stats.norm.ppf((1 + confidence) / 2)
    return EffectSizeResult(
        measure="Pearson r",
        estimate=float(r),
        ci_lower=float(np.tanh(z_r - z_crit * se_z)),
        ci_upper=float(np.tanh(z_r + z_crit * se_z)),
        se=float(se_z),
        n=n,
    )


def r_squared(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
) -> EffectSizeResult:
    """Coefficient of determination *R*-squared.

    Parameters
    ----------
    x, y : array-like

    Returns
    -------
    EffectSizeResult
    """
    r_res = r_effect_size(x, y)
    r2 = r_res.estimate**2
    return EffectSizeResult(
        measure="R-squared",
        estimate=float(r2),
        ci_lower=float(r_res.ci_lower**2) if r_res.ci_lower else None,
        ci_upper=float(r_res.ci_upper**2) if r_res.ci_upper else None,
        n=r_res.n,
    )


def eta_squared(
    ss_effect: float,
    ss_total: float,
) -> EffectSizeResult:
    """Eta-squared from ANOVA sums of squares.

    :math:`\\eta^2 = SS_{effect} / SS_{total}`

    Parameters
    ----------
    ss_effect : float
        Sum of squares for the effect.
    ss_total : float
        Total sum of squares.

    Returns
    -------
    EffectSizeResult
    """
    eta2 = ss_effect / ss_total if ss_total > 0 else 0.0
    return EffectSizeResult(
        measure="Eta-squared",
        estimate=float(eta2),
    )


def partial_eta_squared(
    ss_effect: float,
    ss_error: float,
) -> EffectSizeResult:
    """Partial eta-squared.

    :math:`\\eta^2_p = SS_{effect} / (SS_{effect} + SS_{error})`

    Parameters
    ----------
    ss_effect, ss_error : float

    Returns
    -------
    EffectSizeResult
    """
    denom = ss_effect + ss_error
    pe2 = ss_effect / denom if denom > 0 else 0.0
    return EffectSizeResult(
        measure="Partial eta-squared",
        estimate=float(pe2),
    )


def omega_squared(
    ss_effect: float,
    ss_total: float,
    df_effect: int,
    ms_error: float,
) -> EffectSizeResult:
    """Omega-squared -- less biased than eta-squared.

    :math:`\\omega^2 = (SS_{effect} - df_{effect} \\cdot MS_{error}) / (SS_{total} + MS_{error})`

    Parameters
    ----------
    ss_effect, ss_total : float
    df_effect : int
    ms_error : float

    Returns
    -------
    EffectSizeResult
    """
    num = ss_effect - df_effect * ms_error
    denom = ss_total + ms_error
    w2 = max(num / denom, 0.0) if denom > 0 else 0.0
    return EffectSizeResult(
        measure="Omega-squared",
        estimate=float(w2),
    )


def epsilon_squared(
    ss_effect: float,
    ss_total: float,
    df_effect: int,
    ms_error: float,
) -> EffectSizeResult:
    """Epsilon-squared (Kelley, 1935).

    :math:`\\varepsilon^2 = (SS_{effect} - df_{effect} \\cdot MS_{error}) / SS_{total}`

    Parameters
    ----------
    ss_effect, ss_total : float
    df_effect : int
    ms_error : float

    Returns
    -------
    EffectSizeResult
    """
    num = ss_effect - df_effect * ms_error
    eps2 = max(num / ss_total, 0.0) if ss_total > 0 else 0.0
    return EffectSizeResult(
        measure="Epsilon-squared",
        estimate=float(eps2),
    )


# ===================================================================
# CONTINGENCY TABLE EFFECT SIZES
# ===================================================================


def odds_ratio(
    a: int,
    b: int,
    c: int,
    d: int,
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Odds ratio for a 2x2 table [[a, b], [c, d]].

    :math:`OR = (a \\cdot d) / (b \\cdot c)`

    Parameters
    ----------
    a, b, c, d : int
        Cell counts of the 2x2 table.
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    or_val = (a * d) / (b * c) if b * c > 0 else np.inf
    log_or = math.log(or_val) if or_val > 0 and np.isfinite(or_val) else 0.0
    se_log = math.sqrt(1 / max(a, 1) + 1 / max(b, 1) + 1 / max(c, 1) + 1 / max(d, 1))
    z = stats.norm.ppf((1 + confidence) / 2)
    return EffectSizeResult(
        measure="Odds ratio",
        estimate=float(or_val),
        ci_lower=float(math.exp(log_or - z * se_log)),
        ci_upper=float(math.exp(log_or + z * se_log)),
        se=float(se_log),
        n=a + b + c + d,
        extra={"log_or": log_or},
    )


def risk_ratio(
    a: int,
    b: int,
    c: int,
    d: int,
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Risk ratio (relative risk) for a 2x2 table.

    :math:`RR = [a/(a+b)] / [c/(c+d)]`

    Parameters
    ----------
    a, b, c, d : int
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    p1 = a / (a + b) if (a + b) > 0 else 0.0
    p2 = c / (c + d) if (c + d) > 0 else 0.0
    rr = p1 / p2 if p2 > 0 else np.inf
    log_rr = math.log(rr) if rr > 0 and np.isfinite(rr) else 0.0
    se_log = math.sqrt(b / (a * (a + b)) + d / (c * (c + d))) if a > 0 and c > 0 else np.inf
    z = stats.norm.ppf((1 + confidence) / 2)
    return EffectSizeResult(
        measure="Risk ratio",
        estimate=float(rr),
        ci_lower=float(math.exp(log_rr - z * se_log)),
        ci_upper=float(math.exp(log_rr + z * se_log)),
        se=float(se_log),
        n=a + b + c + d,
    )


def risk_difference(
    a: int,
    b: int,
    c: int,
    d: int,
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Risk difference (attributable risk) for a 2x2 table.

    :math:`RD = a/(a+b) - c/(c+d)`

    Parameters
    ----------
    a, b, c, d : int
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    n1, n2 = a + b, c + d
    p1 = a / n1 if n1 > 0 else 0.0
    p2 = c / n2 if n2 > 0 else 0.0
    rd = p1 - p2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) if n1 > 0 and n2 > 0 else 0.0
    z = stats.norm.ppf((1 + confidence) / 2)
    return EffectSizeResult(
        measure="Risk difference",
        estimate=float(rd),
        ci_lower=float(rd - z * se),
        ci_upper=float(rd + z * se),
        se=float(se),
        n=n1 + n2,
    )


def number_needed_to_treat(
    a: int,
    b: int,
    c: int,
    d: int,
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Number needed to treat (NNT).

    :math:`NNT = 1 / |RD|`

    Parameters
    ----------
    a, b, c, d : int
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    rd_res = risk_difference(a, b, c, d, confidence)
    rd = rd_res.estimate
    nnt = 1 / abs(rd) if abs(rd) > 0 else np.inf
    ci_lo = 1 / abs(rd_res.ci_upper) if rd_res.ci_upper and abs(rd_res.ci_upper) > 0 else np.inf
    ci_hi = 1 / abs(rd_res.ci_lower) if rd_res.ci_lower and abs(rd_res.ci_lower) > 0 else np.inf
    return EffectSizeResult(
        measure="NNT",
        estimate=float(nnt),
        ci_lower=float(min(ci_lo, ci_hi)),
        ci_upper=float(max(ci_lo, ci_hi)),
        n=rd_res.n,
    )


def number_needed_to_harm(
    a: int,
    b: int,
    c: int,
    d: int,
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Number needed to harm (NNH) -- same as NNT but with reversed sign
    convention.

    Parameters
    ----------
    a, b, c, d : int
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    result = number_needed_to_treat(a, b, c, d, confidence)
    return EffectSizeResult(
        measure="NNH",
        estimate=result.estimate,
        ci_lower=result.ci_lower,
        ci_upper=result.ci_upper,
        n=result.n,
    )


def rate_ratio(
    events1: int,
    person_time1: float,
    events2: int,
    person_time2: float,
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Incidence rate ratio.

    :math:`IRR = (e_1/PT_1) / (e_2/PT_2)`

    Parameters
    ----------
    events1, person_time1 : int, float
        Events and person-time in group 1.
    events2, person_time2 : int, float
        Events and person-time in group 2.
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    r1 = events1 / person_time1 if person_time1 > 0 else 0.0
    r2 = events2 / person_time2 if person_time2 > 0 else 0.0
    irr = r1 / r2 if r2 > 0 else np.inf
    log_irr = math.log(irr) if irr > 0 and np.isfinite(irr) else 0.0
    se = math.sqrt(1 / max(events1, 1) + 1 / max(events2, 1))
    z = stats.norm.ppf((1 + confidence) / 2)
    return EffectSizeResult(
        measure="Rate ratio",
        estimate=float(irr),
        ci_lower=float(math.exp(log_irr - z * se)),
        ci_upper=float(math.exp(log_irr + z * se)),
        se=float(se),
        n=events1 + events2,
    )


def incidence_rate_difference(
    events1: int,
    person_time1: float,
    events2: int,
    person_time2: float,
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Incidence rate difference.

    :math:`IRD = (e_1/PT_1) - (e_2/PT_2)`

    Parameters
    ----------
    events1, person_time1 : int, float
    events2, person_time2 : int, float
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    r1 = events1 / person_time1 if person_time1 > 0 else 0.0
    r2 = events2 / person_time2 if person_time2 > 0 else 0.0
    ird = r1 - r2
    se = (
        math.sqrt(events1 / person_time1**2 + events2 / person_time2**2)
        if person_time1 > 0 and person_time2 > 0
        else 0.0
    )
    z = stats.norm.ppf((1 + confidence) / 2)
    return EffectSizeResult(
        measure="Incidence rate difference",
        estimate=float(ird),
        ci_lower=float(ird - z * se),
        ci_upper=float(ird + z * se),
        se=float(se),
    )


# ===================================================================
# ASSOCIATION MEASURES
# ===================================================================


def cohens_w(
    observed: Union[np.ndarray, list],
    expected: Union[np.ndarray, list] | None = None,
) -> EffectSizeResult:
    """Cohen's *w* for chi-squared.

    :math:`w = \\sqrt{\\chi^2 / N}`

    Parameters
    ----------
    observed : array-like
        Observed frequencies.
    expected : array-like or None
        Expected frequencies (uniform if ``None``).

    Returns
    -------
    EffectSizeResult
    """
    obs = np.asarray(observed, dtype=np.float64)
    if expected is None:
        exp = np.full_like(obs, obs.sum() / len(obs))
    else:
        exp = np.asarray(expected, dtype=np.float64)
    n = obs.sum()
    chi2 = np.sum((obs - exp) ** 2 / (exp + 1e-15))
    w = math.sqrt(chi2 / n) if n > 0 else 0.0
    return EffectSizeResult(
        measure="Cohen's w",
        estimate=float(w),
        n=int(n),
    )


def cohens_f(
    eta2: float,
) -> EffectSizeResult:
    """Cohen's *f* from eta-squared.

    :math:`f = \\sqrt{\\eta^2 / (1 - \\eta^2)}`

    Parameters
    ----------
    eta2 : float
        Eta-squared value.

    Returns
    -------
    EffectSizeResult
    """
    f_val = math.sqrt(eta2 / (1 - eta2)) if eta2 < 1 else np.inf
    return EffectSizeResult(
        measure="Cohen's f",
        estimate=float(f_val),
    )


def cramers_v(
    contingency_table: Union[np.ndarray, pd.DataFrame],
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Cramer's *V* for a contingency table.

    :math:`V = \\sqrt{\\chi^2 / (N \\cdot (k - 1))}` where :math:`k = \\min(r, c)`.

    Parameters
    ----------
    contingency_table : array-like or DataFrame
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    table = np.asarray(contingency_table, dtype=np.float64)
    chi2, _, _, _ = stats.chi2_contingency(table, correction=False)
    n = table.sum()
    k = min(table.shape) - 1
    v = math.sqrt(chi2 / (n * k)) if n * k > 0 else 0.0
    # Bias-corrected V
    r, c = table.shape
    v_bc = max(0, v**2 - (k) * (table.shape[0] - 1) / (n - 1))
    v_bc = math.sqrt(v_bc) if v_bc > 0 else 0.0
    return EffectSizeResult(
        measure="Cramer's V",
        estimate=float(v),
        n=int(n),
        extra={"bias_corrected_v": float(v_bc)},
    )


def phi_coefficient(
    contingency_table: Union[np.ndarray, pd.DataFrame],
) -> EffectSizeResult:
    """Phi coefficient for a 2x2 contingency table.

    :math:`\\phi = \\sqrt{\\chi^2 / N}`

    Parameters
    ----------
    contingency_table : array-like

    Returns
    -------
    EffectSizeResult
    """
    table = np.asarray(contingency_table, dtype=np.float64)
    if table.shape != (2, 2):
        raise ValueError("Phi requires a 2x2 table.")
    chi2, _, _, _ = stats.chi2_contingency(table, correction=False)
    n = table.sum()
    phi = math.sqrt(chi2 / n) if n > 0 else 0.0
    # Sign from the OR direction
    if table[0, 0] * table[1, 1] < table[0, 1] * table[1, 0]:
        phi = -phi
    return EffectSizeResult(
        measure="Phi coefficient",
        estimate=float(phi),
        n=int(n),
    )


# ===================================================================
# NON-PARAMETRIC EFFECT SIZES
# ===================================================================


def rank_biserial_correlation(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Rank-biserial correlation (matched rank version).

    :math:`r = 1 - 2U / (n_1 n_2)` where *U* is the Mann--Whitney statistic.

    Parameters
    ----------
    x, y : array-like
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    x, y = _arr(x), _arr(y)
    u, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    nx, ny = len(x), len(y)
    r = 1 - 2 * u / (nx * ny) if nx * ny > 0 else 0.0
    se, ci_lo, ci_hi = _bootstrap_ci(
        lambda a, b: 1 - 2 * stats.mannwhitneyu(a, b, alternative="two-sided").statistic / (len(a) * len(b)),
        (x, y),
        confidence=confidence,
    )
    return EffectSizeResult(
        measure="Rank-biserial correlation",
        estimate=float(r),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        se=se,
        n=nx + ny,
    )


def cliffs_delta(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Cliff's delta (non-parametric effect size).

    :math:`\\delta = (\\#(x_i > y_j) - \\#(x_i < y_j)) / (n_x n_y)`

    Parameters
    ----------
    x, y : array-like
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    x, y = _arr(x), _arr(y)
    nx, ny = len(x), len(y)
    greater = 0
    less = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                greater += 1
            elif xi < yj:
                less += 1
    delta = (greater - less) / (nx * ny) if nx * ny > 0 else 0.0

    def _delta(a, b):
        g = sum(1 for ai in a for bj in b if ai > bj)
        l_ = sum(1 for ai in a for bj in b if ai < bj)
        return (g - l_) / (len(a) * len(b))

    se, ci_lo, ci_hi = _bootstrap_ci(_delta, (x, y), confidence=confidence)
    return EffectSizeResult(
        measure="Cliff's delta",
        estimate=float(delta),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        se=se,
        n=nx + ny,
    )


def vargha_delaney_a(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Vargha--Delaney *A* statistic.

    :math:`A = U / (n_1 n_2)` where *U* is the Mann--Whitney statistic
    (probability that a randomly chosen *x* exceeds a randomly chosen *y*).

    Parameters
    ----------
    x, y : array-like
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    x, y = _arr(x), _arr(y)
    u, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    nx, ny = len(x), len(y)
    a_val = u / (nx * ny) if nx * ny > 0 else 0.5
    se, ci_lo, ci_hi = _bootstrap_ci(
        lambda a, b: stats.mannwhitneyu(a, b, alternative="two-sided").statistic / (len(a) * len(b)),
        (x, y),
        confidence=confidence,
    )
    return EffectSizeResult(
        measure="Vargha-Delaney A",
        estimate=float(a_val),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        se=se,
        n=nx + ny,
    )


# ===================================================================
# REGRESSION
# ===================================================================


def standardized_coefficients(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
) -> pd.DataFrame:
    """Compute standardised regression coefficients (beta weights).

    Standardises X and y to zero mean and unit variance before OLS.

    Parameters
    ----------
    X : array-like or DataFrame
        Predictor matrix (n x p).
    y : array-like or Series
        Outcome variable.

    Returns
    -------
    DataFrame
        Columns: ``variable``, ``beta``, ``se``, ``t``, ``p_value``.
    """
    import statsmodels.api as sm

    if isinstance(X, pd.DataFrame):
        names = X.columns.tolist()
        X_arr = X.values.astype(np.float64)
    else:
        X_arr = np.asarray(X, dtype=np.float64)
        names = [f"x{i}" for i in range(X_arr.shape[1])]
    y_arr = np.asarray(y, dtype=np.float64).ravel()

    # Standardise
    X_std = (X_arr - X_arr.mean(axis=0)) / (X_arr.std(axis=0, ddof=1) + 1e-15)
    y_std = (y_arr - y_arr.mean()) / (y_arr.std(ddof=1) + 1e-15)

    model = sm.OLS(y_std, sm.add_constant(X_std)).fit()
    # Skip constant (index 0)
    results = []
    for i, name in enumerate(names):
        results.append(
            {
                "variable": name,
                "beta": float(model.params[i + 1]),
                "se": float(model.bse[i + 1]),
                "t": float(model.tvalues[i + 1]),
                "p_value": float(model.pvalues[i + 1]),
            }
        )
    return pd.DataFrame(results)


def coefficient_of_variation(
    x: Union[np.ndarray, pd.Series, list],
) -> EffectSizeResult:
    """Coefficient of variation (CV).

    :math:`CV = s / \\bar{x}`

    Parameters
    ----------
    x : array-like

    Returns
    -------
    EffectSizeResult
    """
    x = _arr(x)
    mean = x.mean()
    sd = x.std(ddof=1)
    cv = sd / abs(mean) if abs(mean) > 0 else np.inf
    return EffectSizeResult(
        measure="Coefficient of variation",
        estimate=float(cv),
        n=len(x),
    )


def variance_ratio(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Variance ratio (*F*-test for equality of variances).

    Parameters
    ----------
    x, y : array-like
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    x, y = _arr(x), _arr(y)
    v1, v2 = x.var(ddof=1), y.var(ddof=1)
    f_val = v1 / v2 if v2 > 0 else np.inf
    df1, df2 = len(x) - 1, len(y) - 1
    alpha = (1 - confidence) / 2
    ci_lo = f_val / stats.f.ppf(1 - alpha, df1, df2)
    ci_hi = f_val / stats.f.ppf(alpha, df1, df2)
    p_val = 2 * min(stats.f.cdf(f_val, df1, df2), stats.f.sf(f_val, df1, df2))
    return EffectSizeResult(
        measure="Variance ratio (F)",
        estimate=float(f_val),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        n=len(x) + len(y),
        extra={"p_value": float(p_val), "df1": df1, "df2": df2},
    )


# ===================================================================
# CONVERSION FUNCTIONS
# ===================================================================


def d_to_r(d: float, n1: int | None = None, n2: int | None = None) -> float:
    """Convert Cohen's *d* to Pearson *r*.

    :math:`r = d / \\sqrt{d^2 + a}` where :math:`a = (n_1 + n_2)^2 / (n_1 n_2)`
    or *a = 4* when sample sizes are unknown.

    Parameters
    ----------
    d : float
    n1, n2 : int or None

    Returns
    -------
    float
    """
    if n1 is not None and n2 is not None:
        a = (n1 + n2) ** 2 / (n1 * n2)
    else:
        a = 4.0
    return d / math.sqrt(d**2 + a)


def r_to_d(r: float) -> float:
    """Convert Pearson *r* to Cohen's *d*.

    :math:`d = 2r / \\sqrt{1 - r^2}`

    Parameters
    ----------
    r : float

    Returns
    -------
    float
    """
    return 2 * r / math.sqrt(1 - r**2) if abs(r) < 1 else np.inf * np.sign(r)


def or_to_d(or_val: float) -> float:
    """Convert odds ratio to Cohen's *d* (Hasselblad & Hedges, 1995).

    :math:`d = \\log(OR) \\cdot \\sqrt{3} / \\pi`

    Parameters
    ----------
    or_val : float
        Odds ratio.

    Returns
    -------
    float
    """
    return math.log(or_val) * math.sqrt(3) / math.pi if or_val > 0 else 0.0


def d_to_or(d: float) -> float:
    """Convert Cohen's *d* to an odds ratio.

    :math:`OR = \\exp(d \\pi / \\sqrt{3})`

    Parameters
    ----------
    d : float

    Returns
    -------
    float
    """
    return math.exp(d * math.pi / math.sqrt(3))


def or_to_r(or_val: float) -> float:
    """Convert odds ratio to Pearson *r* via *d*."""
    return d_to_r(or_to_d(or_val))


def r_to_or(r: float) -> float:
    """Convert Pearson *r* to odds ratio via *d*."""
    return d_to_or(r_to_d(r))


def d_to_nnt(d: float, base_rate: float = 0.5) -> float:
    """Convert Cohen's *d* to NNT given a base rate.

    Uses the Kraemer & Kupfer (2006) formula:
    :math:`NNT = 1 / (\\Phi(d/2 + \\Phi^{-1}(CER)) - CER)`

    Parameters
    ----------
    d : float
    base_rate : float, default 0.5
        Control event rate.

    Returns
    -------
    float
    """
    z_cer = stats.norm.ppf(base_rate)
    p_treat = stats.norm.cdf(d + z_cer)  # Note: approximate
    rd = p_treat - base_rate
    return 1 / abs(rd) if abs(rd) > 0 else np.inf


# ===================================================================
# META-ANALYSIS
# ===================================================================


def fixed_effects_meta(
    estimates: Union[np.ndarray, list[float]],
    standard_errors: Union[np.ndarray, list[float]],
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Fixed-effects (inverse-variance weighted) meta-analytic pooling.

    Parameters
    ----------
    estimates : array-like
        Effect-size estimates from *k* studies.
    standard_errors : array-like
        Standard errors of the estimates.
    confidence : float, default 0.95

    Returns
    -------
    EffectSizeResult
    """
    theta = np.asarray(estimates, dtype=np.float64)
    se = np.asarray(standard_errors, dtype=np.float64)
    w = 1 / se**2
    pooled = (w * theta).sum() / w.sum()
    pooled_se = math.sqrt(1 / w.sum())
    z = stats.norm.ppf((1 + confidence) / 2)
    # Q statistic
    q = float(((theta - pooled) ** 2 * w).sum())
    k = len(theta)
    p_q = 1 - stats.chi2.cdf(q, k - 1) if k > 1 else 1.0
    return EffectSizeResult(
        measure="Fixed-effects meta-analysis",
        estimate=float(pooled),
        ci_lower=float(pooled - z * pooled_se),
        ci_upper=float(pooled + z * pooled_se),
        se=float(pooled_se),
        n=k,
        extra={"Q": float(q), "Q_p_value": float(p_q)},
    )


def random_effects_meta(
    estimates: Union[np.ndarray, list[float]],
    standard_errors: Union[np.ndarray, list[float]],
    confidence: float = 0.95,
    method: str = "DL",
) -> EffectSizeResult:
    """Random-effects meta-analytic pooling.

    Parameters
    ----------
    estimates : array-like
    standard_errors : array-like
    confidence : float, default 0.95
    method : str, default "DL"
        Tau-squared estimator: ``"DL"`` (DerSimonian--Laird).

    Returns
    -------
    EffectSizeResult

    References
    ----------
    DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials.
    *Controlled Clinical Trials*, 7(3), 177--188.
    """
    theta = np.asarray(estimates, dtype=np.float64)
    se = np.asarray(standard_errors, dtype=np.float64)
    k = len(theta)
    w = 1 / se**2

    # Fixed-effects pooled
    theta_fe = (w * theta).sum() / w.sum()
    Q = float(((theta - theta_fe) ** 2 * w).sum())

    # DerSimonian-Laird tau-squared
    c = w.sum() - (w**2).sum() / w.sum()
    tau2 = max((Q - (k - 1)) / c, 0.0) if c > 0 else 0.0

    # Random-effects weights
    w_re = 1 / (se**2 + tau2)
    pooled = (w_re * theta).sum() / w_re.sum()
    pooled_se = math.sqrt(1 / w_re.sum())
    z = stats.norm.ppf((1 + confidence) / 2)

    # I-squared
    i2 = max((Q - (k - 1)) / Q, 0.0) * 100 if Q > 0 else 0.0

    # Prediction interval
    pred_se = math.sqrt(pooled_se**2 + tau2)
    t_crit = stats.t.ppf((1 + confidence) / 2, max(k - 2, 1))
    pred_lo = pooled - t_crit * pred_se
    pred_hi = pooled + t_crit * pred_se

    return EffectSizeResult(
        measure="Random-effects meta-analysis (DL)",
        estimate=float(pooled),
        ci_lower=float(pooled - z * pooled_se),
        ci_upper=float(pooled + z * pooled_se),
        se=float(pooled_se),
        n=k,
        extra={
            "tau_squared": float(tau2),
            "tau": float(math.sqrt(tau2)),
            "I_squared": float(i2),
            "Q": float(Q),
            "Q_p_value": float(1 - stats.chi2.cdf(Q, k - 1)) if k > 1 else 1.0,
            "prediction_interval_lower": float(pred_lo),
            "prediction_interval_upper": float(pred_hi),
        },
    )


def i_squared(
    estimates: Union[np.ndarray, list[float]],
    standard_errors: Union[np.ndarray, list[float]],
) -> float:
    """Compute Higgins' *I*-squared heterogeneity statistic.

    :math:`I^2 = \\max(0, (Q - (k-1))/Q) \\times 100`

    Parameters
    ----------
    estimates : array-like
    standard_errors : array-like

    Returns
    -------
    float
        Percentage (0--100).
    """
    result = random_effects_meta(estimates, standard_errors)
    return result.extra.get("I_squared", 0.0)


def prediction_interval(
    estimates: Union[np.ndarray, list[float]],
    standard_errors: Union[np.ndarray, list[float]],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Prediction interval for a new study from a random-effects meta-analysis.

    Parameters
    ----------
    estimates : array-like
    standard_errors : array-like
    confidence : float, default 0.95

    Returns
    -------
    tuple[float, float]
        (lower, upper) prediction interval bounds.
    """
    result = random_effects_meta(estimates, standard_errors, confidence=confidence)
    return (
        result.extra["prediction_interval_lower"],
        result.extra["prediction_interval_upper"],
    )


# ===================================================================
# BOOTSTRAP CI FOR ARBITRARY EFFECT SIZE
# ===================================================================


def bootstrap_effect_size_ci(
    func: callable,
    *arrays: Union[np.ndarray, pd.Series, list],
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> EffectSizeResult:
    """Generic bootstrap CI wrapper for any effect-size function.

    Parameters
    ----------
    func : callable
        Function that takes one or more arrays and returns a scalar.
    *arrays : array-like
        Input arrays to bootstrap.
    n_boot : int, default 2000
    confidence : float, default 0.95
    seed : int, default 42

    Returns
    -------
    EffectSizeResult
    """
    arrs = tuple(_arr(a) for a in arrays)
    point = float(func(*arrs))
    se, ci_lo, ci_hi = _bootstrap_ci(func, arrs, n_boot=n_boot, confidence=confidence, seed=seed)
    return EffectSizeResult(
        measure=f"Bootstrap ({func.__name__})",
        estimate=point,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        se=se,
        n=sum(len(a) for a in arrs),
    )
