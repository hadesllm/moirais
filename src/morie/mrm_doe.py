# SPDX-License-Identifier: GPL-2.0-only
"""Design-of-Experiments (DOE) toolkit.

Closes the Chapter-3/4/5 coverage gap from designexptr.org:
* randomised complete block design (RCBD),
* Latin / Graeco-Latin square,
* fractional 2^(k-p) factorial,
* response surface (Box-Wilson, central composite),
* ANOVA power + Monte-Carlo power,
* block-permutation test,
* random Latin-square generation,
* Bonferroni post-hoc ANOVA.

Primary references:
    Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005).
        Statistics for Experimenters (2nd ed.). Wiley.
    Cochran, W. G., & Cox, G. M. (1957). Experimental Designs (2nd ed.). Wiley.
    Montgomery, D. C. (2017). Design and Analysis of Experiments (9th ed.).
        Wiley.
    Box, G. E. P., & Wilson, K. B. (1951). On the experimental attainment of
        optimum conditions. JRSS-B, 13(1), 1-45.  (Response surface origin.)
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
        (Power machinery.)

Public callables:
    mrm_anova_bonferroni(data, response, group)
    mrm_rcbd(data, response, treatment, block)
    mrm_latin_square(data, response, row, col, treatment)
    mrm_graeco_latin(data, response, row, col, latin, greek)
    mrm_fractional_factorial(data, response, factor_cols, generator)
    mrm_response_surface(data, response, factor_cols)
    mrm_anova_power(k_groups, n_per_group, effect_size_f, alpha)
    mrm_mc_power(simulator, n_sims, alpha)
    mrm_perm_block(data, response, treatment, block, n_perm)
    mrm_random_latin(k, seed)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


__all__ = [
    "mrm_anova_bonferroni",
    "mrm_rcbd",
    "mrm_latin_square",
    "mrm_graeco_latin",
    "mrm_fractional_factorial",
    "mrm_response_surface",
    "mrm_anova_power",
    "mrm_mc_power",
    "mrm_perm_block",
    "mrm_random_latin",
]


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _ols_anova(formula_terms: list[tuple[str, np.ndarray]], y: np.ndarray):
    """Type-I sequential ANOVA from a list of (name, design-matrix) blocks.

    Used internally for RCBD / Latin / Graeco-Latin so we don't drag
    statsmodels in as a hard dep.  Returns list of dicts:
    [{name, df, ss, ms, F, p_value}, ..., {name='Residual', ...}].
    """
    n = y.size
    Xs = []
    rows = []
    Xfull = np.ones((n, 1))  # intercept
    df_total = n - 1
    ss_total = float(np.sum((y - y.mean()) ** 2))
    last_resid_ss = ss_total
    last_resid_df = df_total
    accumulated_X = Xfull.copy()
    for name, Xb in formula_terms:
        new_X = np.column_stack([accumulated_X, Xb])
        # Compute SS reduction by adding this block
        beta_old, *_ = np.linalg.lstsq(accumulated_X, y, rcond=None)
        rss_old = float(np.sum((y - accumulated_X @ beta_old) ** 2))
        beta_new, *_ = np.linalg.lstsq(new_X, y, rcond=None)
        rss_new = float(np.sum((y - new_X @ beta_new) ** 2))
        df_b = Xb.shape[1]
        ss_b = rss_old - rss_new
        ms_b = ss_b / df_b if df_b > 0 else float("nan")
        rows.append({"name": name, "df": df_b, "ss": round(ss_b, 6),
                     "ms": round(ms_b, 6)})
        accumulated_X = new_X
        last_resid_ss = rss_new
        last_resid_df = n - accumulated_X.shape[1]

    ms_res = last_resid_ss / last_resid_df if last_resid_df > 0 else float("nan")
    rows.append({"name": "Residual", "df": int(last_resid_df),
                 "ss": round(last_resid_ss, 6), "ms": round(ms_res, 6)})
    # Compute F and p for each non-residual block
    for r in rows[:-1]:
        if ms_res > 0:
            F = r["ms"] / ms_res
            p = 1 - stats.f.cdf(F, r["df"], last_resid_df)
            r["F"] = round(float(F), 4)
            r["p_value"] = float(p)
        else:
            r["F"] = float("nan")
            r["p_value"] = float("nan")
    rows[-1]["F"] = None
    rows[-1]["p_value"] = None
    return rows


def _factor_dummies(s: pd.Series) -> np.ndarray:
    """Deviation-coded dummies, one fewer column than n_levels."""
    levels = list(pd.unique(s.dropna()))
    if len(levels) < 2:
        return np.zeros((len(s), 0))
    D = np.zeros((len(s), len(levels) - 1))
    for i, lv in enumerate(levels[:-1]):
        D[:, i] = (s.values == lv).astype(float) - (s.values == levels[-1]).astype(float)
    return D


# ─── ANOVA with Bonferroni post-hoc ──────────────────────────────────────────


def mrm_anova_bonferroni(
    data: pd.DataFrame, *,
    response_col: str, group_col: str,
    alpha: float = 0.05,
) -> dict:
    """One-way ANOVA with pairwise Bonferroni-adjusted t-tests.

    Returns the F-test, all pairwise means / mean-differences, the raw
    pairwise t-test p-values, and the Bonferroni-adjusted significance
    flags at level alpha.
    """
    d = data[[response_col, group_col]].dropna()
    groups = list(pd.unique(d[group_col]))
    samples = [d.loc[d[group_col] == g, response_col].to_numpy(dtype=float)
               for g in groups]
    f_stat, p_anova = stats.f_oneway(*samples)
    pairs = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            t, p = stats.ttest_ind(samples[i], samples[j], equal_var=False)
            pairs.append({
                "group_a": groups[i], "group_b": groups[j],
                "diff": round(float(np.mean(samples[i]) - np.mean(samples[j])), 4),
                "t": round(float(t), 4), "p_raw": float(p),
            })
    n_pairs = len(pairs)
    for p in pairs:
        p["p_bonferroni"] = min(1.0, p["p_raw"] * n_pairs)
        p["significant"] = p["p_bonferroni"] < alpha
    return {
        "f_statistic": round(float(f_stat), 4),
        "p_value": float(p_anova),
        "n_groups": int(len(groups)),
        "n_pairs": int(n_pairs),
        "alpha": alpha,
        "alpha_per_pair": alpha / max(n_pairs, 1),
        "pairs": pd.DataFrame(pairs),
        "interpretation": (
            f"F = {f_stat:.3f}, p = {p_anova:.3g}; {sum(p['significant'] for p in pairs)}/{n_pairs} "
            f"pairs significant after Bonferroni at alpha = {alpha}."
        ),
    }


# ─── Randomised complete block design ────────────────────────────────────────


def mrm_rcbd(
    data: pd.DataFrame, *,
    response_col: str, treatment_col: str, block_col: str,
) -> dict:
    """Randomised complete block design (RCBD) two-way ANOVA.

    Model: y_{ij} = mu + tau_i (treatment) + beta_j (block) + eps_{ij}
    Returns Type-I sequential ANOVA: block enters first, then treatment.
    """
    d = data[[response_col, treatment_col, block_col]].dropna()
    y = d[response_col].to_numpy(dtype=float)
    Xb = _factor_dummies(d[block_col])
    Xt = _factor_dummies(d[treatment_col])
    table = _ols_anova([("Block", Xb), ("Treatment", Xt)], y)
    return {
        "anova": pd.DataFrame(table),
        "n": int(len(d)),
        "n_treatments": int(d[treatment_col].nunique()),
        "n_blocks": int(d[block_col].nunique()),
        "interpretation": (
            f"RCBD on n={len(d)}, {d[treatment_col].nunique()} treatments, "
            f"{d[block_col].nunique()} blocks; "
            f"Treatment p = {table[1]['p_value']:.3g}."
        ),
    }


# ─── Latin square ────────────────────────────────────────────────────────────


def mrm_latin_square(
    data: pd.DataFrame, *,
    response_col: str, row_col: str, col_col: str, treatment_col: str,
) -> dict:
    """Latin-square three-way ANOVA (row, column, treatment).

    Eliminates two nuisance sources simultaneously.  Requires
    n_treatments == n_rows == n_cols (the square condition).
    """
    d = data[[response_col, row_col, col_col, treatment_col]].dropna()
    y = d[response_col].to_numpy(dtype=float)
    Xr = _factor_dummies(d[row_col])
    Xc = _factor_dummies(d[col_col])
    Xt = _factor_dummies(d[treatment_col])
    table = _ols_anova([("Row", Xr), ("Column", Xc), ("Treatment", Xt)], y)
    k = d[treatment_col].nunique()
    return {
        "anova": pd.DataFrame(table),
        "n": int(len(d)),
        "k": int(k),
        "interpretation": (
            f"{k}x{k} Latin square on n={len(d)}; Treatment p = {table[2]['p_value']:.3g}."
        ),
    }


# ─── Graeco-Latin square ─────────────────────────────────────────────────────


def mrm_graeco_latin(
    data: pd.DataFrame, *,
    response_col: str, row_col: str, col_col: str,
    latin_col: str, greek_col: str,
) -> dict:
    """Graeco-Latin square four-way ANOVA (row, col, Latin, Greek)."""
    d = data[[response_col, row_col, col_col, latin_col, greek_col]].dropna()
    y = d[response_col].to_numpy(dtype=float)
    Xr = _factor_dummies(d[row_col])
    Xc = _factor_dummies(d[col_col])
    Xl = _factor_dummies(d[latin_col])
    Xg = _factor_dummies(d[greek_col])
    table = _ols_anova([("Row", Xr), ("Column", Xc),
                         ("Latin", Xl), ("Greek", Xg)], y)
    return {
        "anova": pd.DataFrame(table),
        "n": int(len(d)),
        "interpretation": (
            f"Graeco-Latin square on n={len(d)}; "
            f"Latin p = {table[2]['p_value']:.3g}, Greek p = {table[3]['p_value']:.3g}."
        ),
    }


# ─── Fractional factorial 2^(k-p) ────────────────────────────────────────────


def mrm_fractional_factorial(
    data: pd.DataFrame, *,
    response_col: str, factor_cols: Sequence[str],
    generator: Optional[str] = None,
) -> dict:
    """Fractional 2^(k-p) factorial main-effects + alias-structure analysis.

    Assumes factor columns are coded +/-1.  Computes main effects (as
    Yates contrasts) and reports the alias structure if `generator` is
    given (e.g. "D=ABC" means effect D is aliased with ABC).

    Generator string format: comma-separated "X=YZ..." pairs.
    """
    d = data[[response_col, *factor_cols]].dropna()
    y = d[response_col].to_numpy(dtype=float)
    X = d[list(factor_cols)].to_numpy(dtype=float)
    main = {}
    for i, f in enumerate(factor_cols):
        main[f] = float(np.mean(y[X[:, i] == 1]) - np.mean(y[X[:, i] == -1]))
    aliases = {}
    if generator:
        for clause in generator.split(","):
            lhs, rhs = (s.strip() for s in clause.split("="))
            aliases[lhs] = rhs
    return {
        "main_effects": {k: round(v, 6) for k, v in main.items()},
        "alias_structure": aliases,
        "n": int(len(d)),
        "k": int(len(factor_cols)),
        "interpretation": (
            f"2^{len(factor_cols)} fractional on n={len(d)}. Largest main effect: "
            f"{max(main, key=lambda x: abs(main[x]))} = "
            f"{main[max(main, key=lambda x: abs(main[x]))]:.3f}"
        ),
    }


# ─── Response surface (Box-Wilson) ───────────────────────────────────────────


def mrm_response_surface(
    data: pd.DataFrame, *,
    response_col: str, factor_cols: Sequence[str],
) -> dict:
    """Second-order response-surface fit (Box-Wilson 1951).

    Fits y = beta0 + sum beta_i x_i + sum beta_ii x_i^2 + sum beta_ij x_i x_j
    by OLS and returns the stationary point x* = -0.5 * B_inv @ b (if
    invertible), where B is the symmetric quadratic-term matrix and b
    is the linear-term vector.
    """
    d = data[[response_col, *factor_cols]].dropna()
    y = d[response_col].to_numpy(dtype=float)
    X = d[list(factor_cols)].to_numpy(dtype=float)
    k = len(factor_cols)
    # Design columns: intercept, linear, pure-quadratic, cross-products
    cols = [np.ones(len(d))]
    names = ["intercept"]
    for i in range(k):
        cols.append(X[:, i]); names.append(f"{factor_cols[i]}")
    for i in range(k):
        cols.append(X[:, i] ** 2); names.append(f"{factor_cols[i]}^2")
    for i in range(k):
        for j in range(i + 1, k):
            cols.append(X[:, i] * X[:, j])
            names.append(f"{factor_cols[i]}:{factor_cols[j]}")
    D = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    # Build B matrix (symmetric quadratic) and b (linear)
    b = beta[1:1 + k]
    B = np.zeros((k, k))
    for i in range(k):
        B[i, i] = beta[1 + k + i]
    idx = 1 + 2 * k
    for i in range(k):
        for j in range(i + 1, k):
            B[i, j] = beta[idx] / 2; B[j, i] = beta[idx] / 2
            idx += 1
    try:
        x_star = -0.5 * np.linalg.solve(B, b)
        y_star = float(beta[0] + b @ x_star + x_star @ B @ x_star)
    except np.linalg.LinAlgError:
        x_star = None; y_star = float("nan")

    eigvals, _ = np.linalg.eigh(B)
    nature = (
        "maximum" if np.all(eigvals < 0) else
        "minimum" if np.all(eigvals > 0) else
        "saddle"
    )
    return {
        "coefficients": {n: round(float(c), 6) for n, c in zip(names, beta)},
        "stationary_point": (
            None if x_star is None else
            {factor_cols[i]: round(float(x_star[i]), 4) for i in range(k)}
        ),
        "stationary_y": round(y_star, 4) if x_star is not None else None,
        "stationary_nature": nature,
        "eigenvalues": [round(float(v), 4) for v in eigvals],
        "n": int(len(d)),
        "interpretation": (
            f"Second-order RSM on n={len(d)}, k={k}; stationary point is a {nature}."
        ),
    }


# ─── ANOVA power (Cohen f) ───────────────────────────────────────────────────


def mrm_anova_power(
    k_groups: int, n_per_group: int, effect_size_f: float,
    *, alpha: float = 0.05,
) -> dict:
    """Power of one-way ANOVA given Cohen's f effect size.

    Power = P(F > F_crit | non-central F with ncp = N * f^2).
    """
    df1 = k_groups - 1
    N = k_groups * n_per_group
    df2 = N - k_groups
    ncp = N * effect_size_f ** 2
    F_crit = stats.f.ppf(1 - alpha, df1, df2)
    power = 1 - stats.ncf.cdf(F_crit, df1, df2, ncp)
    return {
        "k_groups": int(k_groups),
        "n_per_group": int(n_per_group),
        "N_total": int(N),
        "effect_size_f": float(effect_size_f),
        "alpha": alpha,
        "df1": int(df1), "df2": int(df2),
        "noncentrality": round(float(ncp), 4),
        "F_critical": round(float(F_crit), 4),
        "power": round(float(power), 4),
        "interpretation": (
            f"Power = {power:.3f} for k={k_groups}, n_per_group={n_per_group}, "
            f"Cohen's f = {effect_size_f}, alpha = {alpha}."
        ),
    }


# ─── Monte-Carlo power for arbitrary statistical procedures ──────────────────


def mrm_mc_power(
    simulator: Callable[[int], float],
    *, n_sims: int = 1000, alpha: float = 0.05, seed: int = 42,
) -> dict:
    """Empirical power via Monte-Carlo: caller supplies a simulator
    that draws data under the alternative hypothesis and returns a
    p-value.

    Args:
        simulator: callable(rng_seed) -> p_value
        n_sims: number of simulated datasets.
        alpha: type-I error level.

    Returns:
        dict with empirical_power, sd_pwr, ci95_lower, ci95_upper.
    """
    rng = np.random.default_rng(seed)
    p_values = np.empty(n_sims)
    for i in range(n_sims):
        s = int(rng.integers(0, 2 ** 31 - 1))
        p_values[i] = float(simulator(s))
    rejects = p_values < alpha
    pwr = float(rejects.mean())
    se = float(np.sqrt(pwr * (1 - pwr) / n_sims))
    return {
        "n_sims": int(n_sims),
        "alpha": alpha,
        "empirical_power": round(pwr, 4),
        "se": round(se, 4),
        "ci95_lower": round(max(0.0, pwr - 1.96 * se), 4),
        "ci95_upper": round(min(1.0, pwr + 1.96 * se), 4),
        "interpretation": (
            f"Empirical power = {pwr:.3f} (SE {se:.3f}) over {n_sims} sims at alpha={alpha}."
        ),
    }


# ─── Block-permutation test ──────────────────────────────────────────────────


def mrm_perm_block(
    data: pd.DataFrame, *,
    response_col: str, treatment_col: str, block_col: str,
    n_perm: int = 1000, seed: int = 42,
) -> dict:
    """Block-permutation test for treatment effect.

    Randomly permutes treatment labels within each block, recomputes
    the treatment-mean difference, and compares to the observed
    statistic.  Two-sided p-value.
    """
    d = data[[response_col, treatment_col, block_col]].dropna().reset_index(drop=True)
    obs_stat = (
        d.groupby(treatment_col)[response_col].mean().diff().iloc[-1]
    )
    rng = np.random.default_rng(seed)
    perm_stats = np.empty(n_perm)
    for k in range(n_perm):
        permuted = d.copy()
        for b, idx in d.groupby(block_col).groups.items():
            shuffled = rng.permutation(d.loc[idx, treatment_col].values)
            permuted.loc[idx, treatment_col] = shuffled
        perm_stats[k] = (
            permuted.groupby(treatment_col)[response_col].mean().diff().iloc[-1]
        )
    p = float((np.abs(perm_stats) >= abs(obs_stat)).mean())
    return {
        "observed_statistic": round(float(obs_stat), 6),
        "n_perm": int(n_perm),
        "p_value": p,
        "interpretation": (
            f"Block-permutation test: observed diff = {obs_stat:.4f}, "
            f"two-sided p = {p:.3g} over {n_perm} block-shuffles."
        ),
    }


# ─── Random Latin-square generation ──────────────────────────────────────────


def mrm_random_latin(k: int, *, seed: int = 42) -> pd.DataFrame:
    """Generate a random k x k Latin square via row-cycle + permutation.

    Standard algorithm: build the cyclic Latin square, then randomly
    permute rows, columns, and symbols.  Produces a uniform sample
    over a subset of Latin squares (not a uniform sample over all
    Latin squares — exact uniform sampling requires Jacobson-Matthews
    for k > 5).
    """
    rng = np.random.default_rng(seed)
    base = np.array([[(i + j) % k for j in range(k)] for i in range(k)])
    row_perm = rng.permutation(k)
    col_perm = rng.permutation(k)
    sym_perm = rng.permutation(k)
    sq = base[row_perm][:, col_perm]
    sq = np.vectorize(lambda x: int(sym_perm[x]))(sq)
    return pd.DataFrame(sq,
                          index=[f"R{i+1}" for i in range(k)],
                          columns=[f"C{j+1}" for j in range(k)])
