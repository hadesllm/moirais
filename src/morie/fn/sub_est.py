"""Design-based subpopulation (domain) mean estimator."""

import math

import numpy as np
import pandas as pd
import scipy.stats as scipy_stats


def subpopulation_estimate(
    df: pd.DataFrame,
    domain_col: str,
    domain_value,
    outcome_col: str,
    weight_col: str,
) -> dict:
    """
    Design-based estimate for a domain (subpopulation) mean.

    Treats domain membership as a 0/1 indicator and applies the Hajek
    estimator within the full-sample design frame (i.e., does NOT subset
    then reweight, which would ignore the variance contribution from
    domain membership uncertainty).

    Domain mean:

    .. math::

        \\bar{y}_d = \\frac{\\sum_{i: d_i = 1} w_i y_i}{\\sum_{i: d_i = 1} w_i}

    SE via linearisation (Woodruff, 1971):

    .. math::

        \\hat{V}(\\bar{y}_d) \\approx \\frac{1}{N_d^2}
        \\sum_i w_i^2 z_i^2

    where :math:`z_i = d_i (y_i - \\bar{y}_d)` and
    :math:`N_d = \\sum_{i: d_i=1} w_i`.

    :param df: Full sample DataFrame (do NOT pre-filter to domain).
    :param domain_col: Column name identifying the domain.
    :param domain_value: Value of domain_col that identifies the target subpopulation.
    :param outcome_col: Column name of the outcome variable.
    :param weight_col: Column name of survey weights.
    :return: dict with keys ``mean``, ``se``, ``ci_lower``, ``ci_upper``, ``n_domain``.
    :raises ValueError: If required columns are missing.

    References
    ----------
    Woodruff, R. S. (1971). A simple method for approximating the variance of a
        complicated estimate. JASA, 66(334), 411-414.
    Lumley, T. (2010). Complex Surveys. Wiley. (Section 4.2.)
    """
    for col in [domain_col, outcome_col, weight_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    domain_mask = (df[domain_col] == domain_value).values.astype(float)
    y = df[outcome_col].astype(float).values
    w = df[weight_col].astype(float).values

    if np.any(w <= 0):
        raise ValueError("All weights must be > 0.")

    n_domain = int(domain_mask.sum())
    if n_domain == 0:
        raise ValueError(f"No observations found with {domain_col} == {domain_value!r}.")

    N_d = float(np.sum(w * domain_mask))
    y_bar_d = float(np.sum(w * domain_mask * y) / N_d)

    # Woodruff linearisation: z_i = d_i * (y_i - y_bar_d)
    z = domain_mask * (y - y_bar_d)
    var_d = float(np.sum(w**2 * z**2)) / (N_d**2)
    se = math.sqrt(max(0.0, var_d))

    z_crit = float(scipy_stats.norm.ppf(0.975))
    return {
        "mean": float(y_bar_d),
        "se": float(se),
        "ci_lower": float(y_bar_d - z_crit * se),
        "ci_upper": float(y_bar_d + z_crit * se),
        "n_domain": n_domain,
    }


sub_est_fn = subpopulation_estimate


def cheatsheet() -> str:
    return "subpopulation_estimate({}) -> Design-based subpopulation (domain) mean estimator."
