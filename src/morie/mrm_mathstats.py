# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mathematical-statistics / simulation / computation toolkit.

Closes the Chapter-2 coverage gap from designexptr.org/mathematical-
statistics-simulation-and-computation.html.

Primary references:
    Wilks, S. S. (1962). Mathematical Statistics. Wiley.
    Casella, G. & Berger, R. L. (2002). Statistical Inference. Duxbury.
    Lehmann, E. L. & Romano, J. P. (2005). Testing Statistical Hypotheses.

Public callables:
    mrm_oneprop_test(x, n, p0)   -- exact-binomial + Wald one-prop test
    mrm_twoprop_test(x1, n1, x2, n2)
                                  -- chi-square + Fisher exact + Wald
    mrm_var_test(sample, sigma0_sq)
                                  -- chi-square test for σ² (Wilks 1962)
    mrm_qq_plot(sample, dist)    -- Q-Q plot coordinates
    mrm_clt_demo(base_distribution, n_samples, sample_size)
                                  -- Central Limit Theorem demonstrator
    mrm_pit(sample, dist)        -- Probability Integral Transform
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats


__all__ = [
    "mrm_oneprop_test",
    "mrm_twoprop_test",
    "mrm_var_test",
    "mrm_qq_plot",
    "mrm_clt_demo",
    "mrm_pit",
]


# ─── One-proportion test ─────────────────────────────────────────────────────


@dataclass
class OnePropResult:
    p_hat: float
    p0: float
    n: int
    z_wald: float
    p_value_wald: float
    p_value_exact: float
    ci95_wald_lower: float
    ci95_wald_upper: float
    ci95_exact_lower: float       # Clopper-Pearson
    ci95_exact_upper: float
    interpretation: str


def mrm_oneprop_test(
    x: int, n: int, p0: float, *, alpha: float = 0.05,
) -> OnePropResult:
    """One-proportion test (binomial exact + Wald approximation).

    Args:
        x: number of successes.
        n: number of trials.
        p0: null-hypothesis proportion.
        alpha: CI level (default 0.05 -> 95% CI).
    """
    if n <= 0 or x < 0 or x > n:
        raise ValueError("invalid x, n")
    p_hat = x / n
    se_null = float(np.sqrt(p0 * (1 - p0) / n))
    z = (p_hat - p0) / se_null if se_null > 0 else float("nan")
    p_wald = 2 * (1 - stats.norm.cdf(abs(z)))
    p_exact = float(stats.binomtest(x, n, p=p0).pvalue)
    se = float(np.sqrt(p_hat * (1 - p_hat) / n)) if 0 < p_hat < 1 else 0.0
    z_a = stats.norm.ppf(1 - alpha / 2)
    cp = stats.binomtest(x, n).proportion_ci(method="exact")
    return OnePropResult(
        p_hat=round(p_hat, 6),
        p0=round(p0, 6), n=int(n),
        z_wald=round(float(z), 4),
        p_value_wald=float(p_wald), p_value_exact=p_exact,
        ci95_wald_lower=round(max(0.0, p_hat - z_a * se), 6),
        ci95_wald_upper=round(min(1.0, p_hat + z_a * se), 6),
        ci95_exact_lower=round(cp.low, 6),
        ci95_exact_upper=round(cp.high, 6),
        interpretation=(
            f"p̂ = {p_hat:.4f}, H0: p = {p0}; exact p = {p_exact:.3g} "
            f"({'reject' if p_exact < alpha else 'fail to reject'} H0 at α={alpha})."
        ),
    )


# ─── Two-proportion test ─────────────────────────────────────────────────────


@dataclass
class TwoPropResult:
    p1: float
    p2: float
    diff: float
    chi2: float
    df: int
    p_value_chi2: float
    p_value_fisher: float
    z_wald: float
    p_value_wald: float
    ci95_diff_lower: float
    ci95_diff_upper: float
    interpretation: str


def mrm_twoprop_test(
    x1: int, n1: int, x2: int, n2: int, *, alpha: float = 0.05,
) -> TwoPropResult:
    """Two-proportion test (chi-square + Fisher's exact + Wald)."""
    if any(v <= 0 for v in (n1, n2)) or x1 < 0 or x2 < 0:
        raise ValueError("invalid sample sizes / counts")
    p1, p2 = x1 / n1, x2 / n2
    tbl = np.array([[x1, n1 - x1], [x2, n2 - x2]])
    chi2, p_chi2, dof, _ = stats.chi2_contingency(tbl, correction=False)
    p_fisher = float(stats.fisher_exact(tbl, alternative="two-sided")[1])
    # Wald CI for p1 - p2
    se = float(np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2))
    z_w = (p1 - p2) / se if se > 0 else float("nan")
    p_wald = 2 * (1 - stats.norm.cdf(abs(z_w)))
    z_a = stats.norm.ppf(1 - alpha / 2)
    diff = p1 - p2
    return TwoPropResult(
        p1=round(p1, 6), p2=round(p2, 6),
        diff=round(diff, 6),
        chi2=round(float(chi2), 4), df=int(dof),
        p_value_chi2=float(p_chi2), p_value_fisher=p_fisher,
        z_wald=round(float(z_w), 4), p_value_wald=float(p_wald),
        ci95_diff_lower=round(diff - z_a * se, 6),
        ci95_diff_upper=round(diff + z_a * se, 6),
        interpretation=(
            f"p̂₁={p1:.4f}, p̂₂={p2:.4f}; Δ={diff:.4f}; "
            f"χ²({dof})={chi2:.3f}, p_chi²={p_chi2:.3g}, p_Fisher={p_fisher:.3g}."
        ),
    )


# ─── Variance test (chi-square) ──────────────────────────────────────────────


@dataclass
class VarTestResult:
    s_sq: float
    sigma0_sq: float
    chi2_stat: float
    df: int
    p_value_two_sided: float
    p_value_one_sided_greater: float
    p_value_one_sided_less: float
    ci95_lower: float
    ci95_upper: float
    interpretation: str


def mrm_var_test(
    sample: Iterable[float], sigma0_sq: float, *, alpha: float = 0.05,
) -> VarTestResult:
    """Chi-square test for variance σ² = σ₀² (Wilks 1962).

    Args:
        sample: sample data (assumed iid normal).
        sigma0_sq: null hypothesis variance.
    """
    x = np.asarray([v for v in sample if np.isfinite(v)], dtype=float)
    n = x.size
    if n < 2:
        raise ValueError("need >= 2 observations")
    s_sq = float(np.var(x, ddof=1))
    df = n - 1
    stat = df * s_sq / sigma0_sq
    p_lower = float(stats.chi2.cdf(stat, df))
    p_upper = 1 - p_lower
    p_two = 2 * min(p_lower, p_upper)
    lo = df * s_sq / stats.chi2.ppf(1 - alpha / 2, df)
    hi = df * s_sq / stats.chi2.ppf(alpha / 2, df)
    return VarTestResult(
        s_sq=round(s_sq, 6),
        sigma0_sq=round(float(sigma0_sq), 6),
        chi2_stat=round(float(stat), 4), df=int(df),
        p_value_two_sided=float(p_two),
        p_value_one_sided_greater=float(p_upper),
        p_value_one_sided_less=float(p_lower),
        ci95_lower=round(lo, 6),
        ci95_upper=round(hi, 6),
        interpretation=(
            f"s² = {s_sq:.4f}; H0: σ² = {sigma0_sq}; "
            f"χ²({df}) = {stat:.3f}, two-sided p = {p_two:.3g}."
        ),
    )


# ─── Q-Q plot coordinates ────────────────────────────────────────────────────


def mrm_qq_plot(
    sample: Iterable[float], *,
    dist: str = "norm", **dist_kwargs,
) -> pd.DataFrame:
    """Compute Q-Q plot coordinates (theoretical vs empirical quantiles).

    Args:
        sample: sample data.
        dist: scipy.stats distribution name (default "norm").
        **dist_kwargs: distribution parameters (e.g. loc=0, scale=1 for norm).

    Returns:
        DataFrame with theoretical, empirical, and rank columns. Suitable
        for plotting empirical against theoretical.
    """
    x = np.sort(np.asarray([v for v in sample if np.isfinite(v)], dtype=float))
    n = x.size
    if n < 2:
        raise ValueError("need >= 2 observations")
    pdist = getattr(stats, dist)
    # Plotting positions (Blom 1958)
    p = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
    theoretical = pdist.ppf(p, **dist_kwargs)
    return pd.DataFrame({
        "rank": np.arange(1, n + 1),
        "empirical": x,
        "theoretical": theoretical,
        "plotting_position": p,
    })


# ─── Central Limit Theorem demonstrator ──────────────────────────────────────


def mrm_clt_demo(
    base_distribution: str = "uniform",
    *,
    n_samples: int = 1000,
    sample_size: int = 30,
    seed: int = 42,
    **dist_kwargs,
) -> pd.DataFrame:
    """Generate sample means from a base distribution for CLT visualisation.

    For increasing sample_size, the distribution of the sample mean should
    approach Normal(μ_X, σ_X²/n) by the CLT.

    Args:
        base_distribution: scipy.stats distribution name to draw from.
            Default "uniform" (clear non-normal base) is pedagogical.
        n_samples: number of sample means to compute.
        sample_size: size of each sample.
        seed: RNG seed.
        **dist_kwargs: parameters of the base distribution.

    Returns:
        DataFrame with one row per sample mean.
    """
    rng = np.random.default_rng(seed)
    pdist = getattr(stats, base_distribution)
    # scipy doesn't accept .random_state via rvs cleanly; use rng explicitly
    means = []
    for _ in range(n_samples):
        x = pdist.rvs(size=sample_size, random_state=rng, **dist_kwargs)
        means.append(float(np.mean(x)))
    means = np.array(means)
    return pd.DataFrame({
        "sample_index": np.arange(1, n_samples + 1),
        "sample_mean": means,
        "z_score": (means - means.mean()) / means.std(ddof=1),
    })


# ─── Probability Integral Transform ──────────────────────────────────────────


def mrm_pit(sample: Iterable[float], *, dist: str = "norm", **dist_kwargs) -> pd.DataFrame:
    """Apply the Probability Integral Transform to a sample.

    If X ~ F, then F(X) ~ Uniform(0,1). Returns the empirical and
    PIT-transformed values; if the assumed distribution F is correct,
    the transformed values should be ≈ Uniform(0,1).

    Args:
        sample: sample data.
        dist: assumed CDF (scipy.stats distribution name).
        **dist_kwargs: distribution parameters.

    Returns:
        DataFrame with raw and U = F(raw) columns.
    """
    x = np.asarray([v for v in sample if np.isfinite(v)], dtype=float)
    pdist = getattr(stats, dist)
    U = pdist.cdf(x, **dist_kwargs)
    # KS test of U against Uniform(0,1) -- diagnostic for fit quality
    ks = stats.kstest(U, "uniform")
    out = pd.DataFrame({"raw": x, "U": U})
    out.attrs["ks_stat"] = float(ks.statistic)
    out.attrs["ks_pvalue"] = float(ks.pvalue)
    return out
