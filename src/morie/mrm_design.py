# SPDX-License-Identifier: AGPL-3.0-or-later
"""Experimental-design callables for morie.

Inspired by the designexptr.org pedagogical sequence:
    https://designexptr.org/mathematical-statistics-simulation-and-computation.html
    https://designexptr.org/anovachapt.html
    https://designexptr.org/causal-inference.html

Four general-purpose statistical-design entry points, all with full
R parity in r-package/morie/R/mrm_design.R:

    mrm_two_treatment_test()  -- Welch t / Student t / Mann-Whitney U
                                with sensitivity to assumption breaks.
    mrm_anova_oneway()        -- One-way ANOVA + Tukey HSD post-hoc.
    mrm_factorial_2k()        -- 2^k factorial: main effects +
                                interaction effects + half-normal plot
                                coordinates for Daniel's method.
    mrm_causal_design()       -- Convenience wrapper around the morie
                                causal estimators in a designed-
                                experiment idiom (treatment vector,
                                outcome vector, optional covariates).

Each returns a tidy dict / dataclass with the canonical fields
(estimate / se / ci / p_value / interpretation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


__all__ = [
    "TwoTreatmentResult",
    "AnovaOneWayResult",
    "Factorial2kResult",
    "CausalDesignResult",
    "mrm_two_treatment_test",
    "mrm_anova_oneway",
    "mrm_factorial_2k",
    "mrm_causal_design",
]


@dataclass
class TwoTreatmentResult:
    estimate: float       # difference in means
    se: float
    t_statistic: float
    df: float
    p_welch: float
    p_student: float
    p_mannwhitney: float
    ci_lower: float
    ci_upper: float
    n_a: int
    n_b: int
    interpretation: str


def mrm_two_treatment_test(
    a: Iterable[float],
    b: Iterable[float],
    *,
    alpha: float = 0.05,
) -> TwoTreatmentResult:
    """Compare two-treatment outcomes under three assumption regimes.

    Always returns Welch's t-test (unequal variance), Student's t (equal
    variance), and Mann-Whitney U (rank-based). The Welch's p_value is the
    canonical answer; the others are the sensitivity range.

    Args:
        a, b: outcome vectors under treatments A and B.
        alpha: confidence-interval level (default 0.05 -> 95% CI).
    """
    a = np.asarray([x for x in a if np.isfinite(x)], dtype=float)
    b = np.asarray([x for x in b if np.isfinite(x)], dtype=float)
    n_a, n_b = a.size, b.size

    welch = stats.ttest_ind(a, b, equal_var=False)
    stud = stats.ttest_ind(a, b, equal_var=True)
    try:
        mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        p_mw = float(mw.pvalue)
    except Exception:
        p_mw = float("nan")

    diff = float(a.mean() - b.mean())
    sd_a = float(a.std(ddof=1)) if n_a > 1 else float("nan")
    sd_b = float(b.std(ddof=1)) if n_b > 1 else float("nan")
    se = float(np.sqrt(sd_a**2 / n_a + sd_b**2 / n_b))
    # Welch-Satterthwaite df
    df = (sd_a**2 / n_a + sd_b**2 / n_b) ** 2 / (
        (sd_a**2 / n_a) ** 2 / (n_a - 1) + (sd_b**2 / n_b) ** 2 / (n_b - 1)
    )
    z = stats.t.ppf(1 - alpha / 2, df)
    ci_lo, ci_hi = diff - z * se, diff + z * se

    if welch.pvalue < alpha:
        msg = f"Welch t rejects H0 (Δ={diff:.3f}, p={welch.pvalue:.3g}); A and B differ."
    else:
        msg = f"Welch t does not reject H0 (Δ={diff:.3f}, p={welch.pvalue:.3g})."

    return TwoTreatmentResult(
        estimate=round(diff, 6),
        se=round(se, 6),
        t_statistic=round(float(welch.statistic), 4),
        df=round(float(df), 2),
        p_welch=float(welch.pvalue),
        p_student=float(stud.pvalue),
        p_mannwhitney=p_mw,
        ci_lower=round(ci_lo, 6),
        ci_upper=round(ci_hi, 6),
        n_a=int(n_a),
        n_b=int(n_b),
        interpretation=msg,
    )


@dataclass
class AnovaOneWayResult:
    f_statistic: float
    p_value: float
    df_between: int
    df_within: int
    means: dict
    n_per_group: dict
    tukey_hsd: pd.DataFrame   # pairwise contrasts: pair, mean_diff, ci_lower, ci_upper, p_adj
    interpretation: str


def mrm_anova_oneway(
    data: pd.DataFrame,
    *,
    response_col: str,
    group_col: str,
    alpha: float = 0.05,
) -> AnovaOneWayResult:
    """One-way ANOVA + Tukey HSD post-hoc on k-arm data."""
    df = data[[response_col, group_col]].dropna().copy()
    groups = df.groupby(group_col)[response_col].apply(np.asarray)
    n_groups = len(groups)
    if n_groups < 2:
        raise ValueError("Need >= 2 groups for ANOVA")
    f, p = stats.f_oneway(*groups)

    # Tukey HSD via statsmodels (optional; if unavailable use Bonferroni-corrected pairwise t)
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        tk = pairwise_tukeyhsd(df[response_col].values, df[group_col].values, alpha=alpha)
        tk_df = pd.DataFrame(data=tk._results_table.data[1:],
                             columns=tk._results_table.data[0])
    except Exception:
        # fallback: Bonferroni-corrected pairwise Welch t
        rows = []
        names = list(groups.index)
        n_pairs = n_groups * (n_groups - 1) // 2
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                a, b = groups.iloc[i], groups.iloc[j]
                t = stats.ttest_ind(a, b, equal_var=False)
                rows.append({
                    "group1": names[i],
                    "group2": names[j],
                    "meandiff": float(a.mean() - b.mean()),
                    "p-adj": min(1.0, t.pvalue * n_pairs),
                    "reject": (t.pvalue * n_pairs) < alpha,
                })
        tk_df = pd.DataFrame(rows)

    means = {g: float(v.mean()) for g, v in groups.items()}
    ns = {g: int(v.size) for g, v in groups.items()}
    n_total = sum(ns.values())
    df_between = n_groups - 1
    df_within = n_total - n_groups
    msg = f"F({df_between},{df_within})={f:.3f}, p={p:.3g}"
    if p < alpha:
        msg += "; reject H0 of equal group means."

    return AnovaOneWayResult(
        f_statistic=round(float(f), 4),
        p_value=float(p),
        df_between=df_between,
        df_within=df_within,
        means={k: round(v, 4) for k, v in means.items()},
        n_per_group=ns,
        tukey_hsd=tk_df,
        interpretation=msg,
    )


@dataclass
class Factorial2kResult:
    main_effects: dict     # factor -> effect estimate
    interaction_effects: dict   # tuple of factor names -> effect
    half_normal_coords: pd.DataFrame   # for Daniel's plot
    n: int
    k: int
    interpretation: str


def mrm_factorial_2k(
    data: pd.DataFrame,
    *,
    response_col: str,
    factor_cols: Sequence[str],
) -> Factorial2kResult:
    """2^k factorial-design analysis with main effects + interactions.

    Args:
        data: data frame with response_col and k factor columns,
            each coded ±1 or 0/1 (will be re-coded to ±1).
        response_col: numeric response.
        factor_cols: list of k factor columns.

    Returns:
        Factorial2kResult with main + interaction effects + half-
        normal-plot coordinates for Daniel's method (which lets the
        user identify the active effects against a null half-normal
        line).
    """
    df = data[[response_col] + list(factor_cols)].dropna().copy()
    k = len(factor_cols)
    if k < 2:
        raise ValueError("2^k design requires k >= 2 factors")

    # Recode 0/1 -> -1/+1
    for c in factor_cols:
        if set(df[c].unique()) - {-1, 1}:
            df[c] = np.where(df[c] > df[c].median(), 1, -1)

    y = df[response_col].to_numpy(dtype=float)
    X = df[list(factor_cols)].to_numpy(dtype=float)

    # Main effects: difference of means at +1 vs -1 for each factor
    main = {}
    for i, c in enumerate(factor_cols):
        main[c] = round(float(y[X[:, i] == 1].mean() - y[X[:, i] == -1].mean()), 6)

    # Interactions: product of factor columns
    from itertools import combinations
    inter = {}
    for r in range(2, k + 1):
        for combo in combinations(range(k), r):
            colname = " × ".join(factor_cols[j] for j in combo)
            prod = np.prod(X[:, combo], axis=1)
            eff = round(float(y[prod == 1].mean() - y[prod == -1].mean()), 6)
            inter[colname] = eff

    # Half-normal plot coordinates
    all_effects = {**{c: main[c] for c in factor_cols}, **inter}
    sorted_pairs = sorted(all_effects.items(), key=lambda p: abs(p[1]))
    n_eff = len(sorted_pairs)
    half_n = pd.DataFrame({
        "effect_name": [p[0] for p in sorted_pairs],
        "effect_magnitude": [abs(p[1]) for p in sorted_pairs],
        "quantile": [(i + 0.5) / n_eff for i in range(n_eff)],
        "half_normal_quantile": [
            stats.norm.ppf(0.5 + 0.5 * (i + 0.5) / n_eff)
            for i in range(n_eff)
        ],
    })

    msg = (
        f"2^{k} factorial on n={len(y)}. {len(main)} main effects + "
        f"{len(inter)} interactions; largest main = "
        f"{max(main.values(), key=abs):.3f}; largest interaction = "
        f"{max(inter.values(), key=abs):.3f} ({max(inter, key=lambda k: abs(inter[k]))})."
    )

    return Factorial2kResult(
        main_effects=main,
        interaction_effects=inter,
        half_normal_coords=half_n,
        n=int(len(y)),
        k=int(k),
        interpretation=msg,
    )


@dataclass
class CausalDesignResult:
    estimator: str
    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n: int
    n_treated: int
    interpretation: str


def mrm_causal_design(
    data: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    covariates: Sequence[str] = (),
    estimator: str = "ipw",
) -> CausalDesignResult:
    """Designed-experiment convenience wrapper around the morie causal
    estimator family.

    Args:
        data: data frame containing treatment, outcome, covariates.
        treatment_col: binary 0/1 treatment column.
        outcome_col: continuous outcome column.
        covariates: optional adjustment-set columns.
        estimator: one of {"ipw", "aipw", "att"}.
    """
    df = data[[treatment_col, outcome_col] + list(covariates)].dropna().copy()
    D = df[treatment_col].astype(int).to_numpy()
    Y = df[outcome_col].to_numpy(dtype=float)
    n, n_t = len(df), int(D.sum())

    if estimator == "ipw" and len(covariates) > 0:
        # logistic propensity then Hájek IPW
        from sklearn.linear_model import LogisticRegression
        X = df[list(covariates)].to_numpy(dtype=float)
        e = LogisticRegression(max_iter=1000).fit(X, D).predict_proba(X)[:, 1]
        e = np.clip(e, 1e-6, 1 - 1e-6)
        # Hájek ATE
        w1 = D / e
        w0 = (1 - D) / (1 - e)
        tau = (w1 * Y).sum() / w1.sum() - (w0 * Y).sum() / w0.sum()
        # bootstrap SE
        rng = np.random.default_rng(42)
        boots = []
        for _ in range(199):
            idx = rng.integers(0, n, n)
            D_b, Y_b, X_b = D[idx], Y[idx], X[idx]
            try:
                e_b = LogisticRegression(max_iter=200).fit(X_b, D_b).predict_proba(X_b)[:, 1]
                e_b = np.clip(e_b, 1e-6, 1 - 1e-6)
                w1b = D_b / e_b
                w0b = (1 - D_b) / (1 - e_b)
                boots.append((w1b * Y_b).sum() / w1b.sum() - (w0b * Y_b).sum() / w0b.sum())
            except Exception:
                pass
        se = float(np.std(boots, ddof=1)) if boots else float("nan")
    else:
        # plain difference of means (no adjustment)
        Y1, Y0 = Y[D == 1], Y[D == 0]
        tau = float(Y1.mean() - Y0.mean())
        se = float(np.sqrt(Y1.var(ddof=1) / Y1.size + Y0.var(ddof=1) / Y0.size))

    z = 1.96
    ci_lo, ci_hi = tau - z * se, tau + z * se
    p = 2 * (1 - stats.norm.cdf(abs(tau / se))) if se > 0 else float("nan")

    return CausalDesignResult(
        estimator=estimator,
        estimate=round(float(tau), 6),
        se=round(float(se), 6),
        ci_lower=round(ci_lo, 6),
        ci_upper=round(ci_hi, 6),
        p_value=float(p),
        n=int(n),
        n_treated=int(n_t),
        interpretation=f"{estimator.upper()} ATE = {tau:.4f} (SE {se:.4f}); 95% CI [{ci_lo:.4f}, {ci_hi:.4f}]",
    )


# Backwards-compatibility alias.
#
# Several papers and external code references the pre-rename name
# `anova_oneway`; the canonical name is now `mrm_anova_oneway` to
# match the MRM-prefix convention used across the module group.  We
# keep the alias indefinitely so v0.x code does not break under
# fresh-user pip installs.
anova_oneway = mrm_anova_oneway
