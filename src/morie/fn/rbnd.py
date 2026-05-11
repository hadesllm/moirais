# morie.fn — function file (hadesllm/morie)
"""Rosenbaum bounds sensitivity analysis for hidden confounding."""

import math

import numpy as np
import pandas as pd
import scipy.stats as scipy_stats


def sensitivity_rosenbaum(
    data: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    covariates: list[str],
    gamma_range: tuple[float, float] = (1.0, 3.0),
    n_gamma: int = 20,
) -> pd.DataFrame:
    """
    Rosenbaum bounds sensitivity analysis for hidden confounding.

    Tests whether the sign-rank test conclusion is robust to an unmeasured
    confounder that could increase the odds of treatment assignment by a
    factor of Gamma (the sensitivity parameter).

    For a range of Gamma values, the method computes:
    - p_lower: p-value under the most favourable assignment (best case)
    - p_upper: p-value under the most adverse assignment (worst case)

    The analysis is based on the Wilcoxon signed-rank statistic applied to
    matched pairs. Here we approximate the matched analysis by using all
    discordant (T=1, T=0) pairs sorted by outcome.

    :param data: DataFrame containing all required columns.
    :param treatment: Column name of the binary treatment indicator (0/1).
    :param outcome: Column name of the outcome variable.
    :param covariates: Covariate column names (used for matching approximation).
    :param gamma_range: Tuple (min_gamma, max_gamma). Default (1.0, 3.0).
    :param n_gamma: Number of Gamma values to evaluate. Default 20.
    :return: DataFrame with columns ``Gamma``, ``p_lower``, ``p_upper``.
    :raises ValueError: If required columns are missing or gamma_range is invalid.

    Notes
    -----
    This implementation uses the normal approximation to the signed-rank
    distribution under sensitivity bounds (Rosenbaum, 2002, Chapter 4).
    For small samples or exact analysis, use the R ``sensitivitymw`` or
    ``rbounds`` package.

    References
    ----------
    Rosenbaum, P. R. (2002). Observational Studies (2nd ed.). Springer. (Chapter 4.)
    Rosenbaum, P. R. (2007). Sensitivity analysis for m-estimates, tests, and
        confidence intervals in matched observational studies. Biometrics, 63(2), 456-464.
    """
    required_cols = [treatment, outcome]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Columns missing from data: {missing}.")
    if gamma_range[0] < 1.0:
        raise ValueError(f"Minimum Gamma must be >= 1.0, got {gamma_range[0]}.")
    if gamma_range[1] <= gamma_range[0]:
        raise ValueError("gamma_range[1] must be > gamma_range[0].")
    if n_gamma < 2:
        raise ValueError(f"n_gamma must be >= 2, got {n_gamma}.")

    df = data[[treatment, outcome]].dropna().copy()
    treated = df[df[treatment] == 1][outcome].values
    control = df[df[treatment] == 0][outcome].values

    # Pair each treated unit with the nearest-ranked control unit
    # (simplified matching by sorted rank proximity)
    min_n = min(len(treated), len(control))
    if min_n < 2:
        raise ValueError("At least 2 treated and 2 control units are required for Rosenbaum bounds.")
    treated_sorted = np.sort(treated)[:min_n]
    control_sorted = np.sort(control)[:min_n]
    differences = treated_sorted - control_sorted

    # Wilcoxon signed-rank statistic (T+)
    n_pairs = len(differences)
    abs_diff = np.abs(differences)
    ranks = scipy_stats.rankdata(abs_diff)
    T_plus = float(np.sum(ranks[differences > 0]))

    gammas = np.linspace(gamma_range[0], gamma_range[1], n_gamma)
    results = []
    for gamma in gammas:
        # Under Gamma, the maximum p_i = gamma / (1 + gamma) (worst case for T+)
        # and minimum p_i = 1 / (1 + gamma) (best case for T+)
        p_max = gamma / (1.0 + gamma)  # upper bound (worst case H0 rejection)
        p_min = 1.0 / (1.0 + gamma)  # lower bound (best case H0 rejection)

        # Expected value and variance of T+ under each extreme
        mu_upper = n_pairs * (n_pairs + 1) / 2 * p_max
        var_upper = n_pairs * (n_pairs + 1) * (2 * n_pairs + 1) / 6 * p_max * (1 - p_max)

        mu_lower = n_pairs * (n_pairs + 1) / 2 * p_min
        var_lower = n_pairs * (n_pairs + 1) * (2 * n_pairs + 1) / 6 * p_min * (1 - p_min)

        # Two-sided p-values using normal approximation
        if var_upper > 0:
            z_upper = (T_plus - mu_upper) / math.sqrt(var_upper)
            p_upper = 2.0 * float(scipy_stats.norm.sf(abs(z_upper)))
        else:
            p_upper = float("nan")

        if var_lower > 0:
            z_lower = (T_plus - mu_lower) / math.sqrt(var_lower)
            p_lower = 2.0 * float(scipy_stats.norm.sf(abs(z_lower)))
        else:
            p_lower = float("nan")

        results.append(
            {
                "Gamma": float(gamma),
                "p_lower": float(p_lower),
                "p_upper": float(p_upper),
            }
        )

    return pd.DataFrame(results)


rbnd_fn = sensitivity_rosenbaum


def cheatsheet() -> str:
    return "sensitivity_rosenbaum({}) -> Rosenbaum bounds sensitivity analysis for hidden confounding"
