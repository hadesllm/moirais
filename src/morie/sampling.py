"""
Survey sampling, resampling, and design-weight utilities.

This module provides functions for common sampling designs used in
epidemiological and public health research: simple random sampling,
stratified sampling, cluster sampling, probability-proportional-to-size
(PPS) sampling, bootstrap resampling, jackknife variance estimation, and
design-weight computation.

All stochastic functions use :func:`numpy.random.default_rng` for
reproducibility rather than legacy global-state RNGs.

References
----------
Cochran, W. G. (1977). *Sampling Techniques* (3rd ed.). Wiley.

Kish, L. (1965). *Survey Sampling*. Wiley.

Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
Chapman & Hall/CRC.

Wolter, K. M. (2007). *Introduction to Variance Estimation* (2nd ed.).
Springer.

Lumley, T. (2010). *Complex Surveys: A Guide to Analysis Using R*. Wiley.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simple random sampling
# ---------------------------------------------------------------------------


def simple_random_sample(
    df: pd.DataFrame,
    n: int,
    *,
    replace: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Draw a simple random sample (SRS) from a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Population frame.
    n : int
        Number of units to sample.
    replace : bool, optional
        If True, sample with replacement (default False).
    seed : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    pandas.DataFrame
        Sampled rows with their original index preserved.

    Raises
    ------
    ValueError
        If *n* exceeds the number of rows when sampling without replacement.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": range(100)})
    >>> sample = simple_random_sample(df, 10, seed=0)
    >>> len(sample)
    10

    References
    ----------
    Cochran, W. G. (1977). *Sampling Techniques* (3rd ed.), Chapter 2.
    """
    if not replace and n > len(df):
        raise ValueError(f"Cannot draw n={n} without replacement from a frame of size {len(df)}")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(df), size=n, replace=replace)
    return df.iloc[indices].copy()


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


def stratified_sample(
    df: pd.DataFrame,
    strata_col: str,
    n_per_stratum: int | dict[str, int],
    *,
    proportional: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Draw a stratified random sample.

    Supports both fixed allocation (same *n* per stratum) and proportional
    allocation.  When ``proportional=True``, *n_per_stratum* is interpreted
    as the total desired sample size and is allocated proportionally across
    strata.

    Parameters
    ----------
    df : pandas.DataFrame
        Population frame.
    strata_col : str
        Column defining strata.
    n_per_stratum : int or dict[str, int]
        If ``proportional=False``: fixed integer per stratum (or a dict mapping
        stratum values to per-stratum sample sizes).
        If ``proportional=True``: total desired sample size (must be int).
    proportional : bool, optional
        If True, allocate sample sizes proportionally to stratum sizes
        (default False).
    seed : int, optional
        Random seed (default 42).

    Returns
    -------
    pandas.DataFrame
        Sampled rows with original index preserved.

    Raises
    ------
    ValueError
        If a requested stratum size exceeds the stratum population.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"stratum": ["A"]*50 + ["B"]*50, "x": range(100)})
    >>> sample = stratified_sample(df, "stratum", 10, seed=0)
    >>> sample.groupby("stratum").size().to_dict()
    {'A': 10, 'B': 10}

    References
    ----------
    Cochran, W. G. (1977). *Sampling Techniques* (3rd ed.), Chapter 5.
    """
    _warn_missing(df, strata_col, "stratified_sample")
    clean = df.dropna(subset=[strata_col])
    rng = np.random.default_rng(seed)

    groups = clean.groupby(strata_col)

    if proportional:
        if not isinstance(n_per_stratum, int):
            raise TypeError("When proportional=True, n_per_stratum must be an integer total sample size")
        total_n = n_per_stratum
        stratum_sizes = groups.size()
        fractions = stratum_sizes / stratum_sizes.sum()
        allocation = (fractions * total_n).round().astype(int)
        # Adjust rounding: add/remove from largest stratum
        diff = total_n - allocation.sum()
        if diff != 0:
            largest = allocation.idxmax()
            allocation[largest] += diff
        n_map = allocation.to_dict()
    elif isinstance(n_per_stratum, dict):
        n_map = n_per_stratum
    else:
        n_map = {name: n_per_stratum for name in groups.groups.keys()}

    samples = []
    for stratum_val, group_df in groups:
        n_draw = n_map.get(stratum_val, 0)
        if n_draw > len(group_df):
            raise ValueError(f"Stratum '{stratum_val}': requested n={n_draw} exceeds population size {len(group_df)}")
        if n_draw > 0:
            idx = rng.choice(len(group_df), size=n_draw, replace=False)
            samples.append(group_df.iloc[idx])

    return pd.concat(samples).copy() if samples else clean.iloc[:0].copy()


# ---------------------------------------------------------------------------
# Cluster sampling
# ---------------------------------------------------------------------------


def cluster_sample(
    df: pd.DataFrame,
    cluster_col: str,
    n_clusters: int,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Two-stage cluster sampling: select clusters, then take all units within.

    This implements single-stage cluster sampling (select clusters, enumerate
    all units within selected clusters).

    Parameters
    ----------
    df : pandas.DataFrame
        Population frame with a cluster identifier column.
    cluster_col : str
        Column identifying clusters (e.g., PSU, school, clinic).
    n_clusters : int
        Number of clusters to select.
    seed : int, optional
        Random seed (default 42).

    Returns
    -------
    pandas.DataFrame
        All rows belonging to the selected clusters.

    Raises
    ------
    ValueError
        If *n_clusters* exceeds the number of distinct clusters.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "cluster": [1]*10 + [2]*10 + [3]*10,
    ...     "y": range(30),
    ... })
    >>> sample = cluster_sample(df, "cluster", 2, seed=0)
    >>> sample["cluster"].nunique()
    2

    References
    ----------
    Cochran, W. G. (1977). *Sampling Techniques* (3rd ed.), Chapter 9.
    """
    _warn_missing(df, cluster_col, "cluster_sample")
    clean = df.dropna(subset=[cluster_col])
    clusters = clean[cluster_col].unique()

    if n_clusters > len(clusters):
        raise ValueError(f"Requested {n_clusters} clusters but only {len(clusters)} exist")

    rng = np.random.default_rng(seed)
    selected = rng.choice(clusters, size=n_clusters, replace=False)
    return clean[clean[cluster_col].isin(selected)].copy()


# ---------------------------------------------------------------------------
# PPS sampling
# ---------------------------------------------------------------------------


def pps_sample(
    df: pd.DataFrame,
    size_col: str,
    n: int,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Probability-proportional-to-size (PPS) sampling.

    Draws *n* units with probability proportional to the values in
    *size_col*.  Negative or zero size values are excluded with a warning.

    Parameters
    ----------
    df : pandas.DataFrame
        Population frame.
    size_col : str
        Column containing the size measure (must be positive).
    n : int
        Number of units to draw.
    seed : int, optional
        Random seed (default 42).

    Returns
    -------
    pandas.DataFrame
        Sampled rows.

    Raises
    ------
    ValueError
        If no positive size values remain after filtering.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"pop": [1000, 500, 200, 100], "name": ["A","B","C","D"]})
    >>> sample = pps_sample(df, "pop", 2, seed=0)
    >>> len(sample)
    2

    References
    ----------
    Brewer, K. R. W., & Hanif, M. (1983). *Sampling with Unequal
    Probabilities*. Springer.
    """
    _warn_missing(df, size_col, "pps_sample")
    clean = df.dropna(subset=[size_col]).copy()
    mask = clean[size_col] > 0
    if not mask.any():
        raise ValueError(f"No positive values in column '{size_col}'")
    if not mask.all():
        n_dropped = int((~mask).sum())
        warnings.warn(
            f"pps_sample: dropped {n_dropped} rows with non-positive size values",
            stacklevel=2,
        )
        clean = clean[mask]

    sizes = clean[size_col].values.astype(float)
    probs = sizes / sizes.sum()

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(clean), size=n, replace=False, p=probs)
    return clean.iloc[indices].copy()


# ---------------------------------------------------------------------------
# Bootstrap resampling
# ---------------------------------------------------------------------------


def bootstrap_sample(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
    *,
    statistic: Callable[[pd.DataFrame], float],
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap resampling for any scalar statistic.

    Draws *n_bootstrap* resamples with replacement from *df*, applies
    *statistic* to each, and returns the bootstrap distribution with
    percentile confidence interval.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    n_bootstrap : int, optional
        Number of bootstrap replicates (default 1000).
    statistic : callable
        Function ``f(df) -> float`` computing the statistic of interest.
    seed : int, optional
        Random seed (default 42).

    Returns
    -------
    dict
        Keys: ``mean`` (float), ``se`` (float), ``ci_lower`` (float),
        ``ci_upper`` (float), ``distribution`` (numpy.ndarray of length
        *n_bootstrap*).

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"x": np.random.default_rng(0).normal(5, 1, 200)})
    >>> result = bootstrap_sample(df, 500, statistic=lambda d: d["x"].mean(), seed=0)
    >>> 4.5 < result["mean"] < 5.5
    True

    References
    ----------
    Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the
    Bootstrap*. Chapman & Hall/CRC.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    boot_stats = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_stats[b] = statistic(df.iloc[idx])

    mean_val = float(np.mean(boot_stats))
    se_val = float(np.std(boot_stats, ddof=1))
    ci_lower = float(np.percentile(boot_stats, 2.5))
    ci_upper = float(np.percentile(boot_stats, 97.5))

    return {
        "mean": mean_val,
        "se": se_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "distribution": boot_stats,
    }


# ---------------------------------------------------------------------------
# Jackknife variance estimation
# ---------------------------------------------------------------------------


def jackknife_estimate(
    df: pd.DataFrame,
    *,
    statistic: Callable[[pd.DataFrame], float],
) -> dict[str, float]:
    """Delete-1 jackknife variance estimate for a scalar statistic.

    Computes the leave-one-out estimates :math:`\\hat{\\theta}_{(-i)}` and
    derives the jackknife variance estimate:

    .. math::

        \\widehat{\\text{Var}}_J = \\frac{n-1}{n} \\sum_{i=1}^{n}
        \\left(\\hat{\\theta}_{(-i)} - \\bar{\\theta}_{(\\cdot)}\\right)^2

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    statistic : callable
        Function ``f(df) -> float`` computing the statistic of interest.

    Returns
    -------
    dict
        Keys: ``estimate`` (float --- full-sample statistic),
        ``se`` (float --- jackknife standard error),
        ``bias`` (float --- jackknife bias estimate).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    >>> jk = jackknife_estimate(df, statistic=lambda d: d["x"].mean())
    >>> abs(jk["estimate"] - 3.0) < 1e-10
    True

    References
    ----------
    Quenouille, M. H. (1956). Notes on bias in estimation. *Biometrika*,
    43(3/4), 353--360.

    Tukey, J. W. (1958). Bias and confidence in not-quite large samples
    (abstract). *Annals of Mathematical Statistics*, 29, 614.
    """
    n = len(df)
    theta_full = statistic(df)
    theta_loo = np.empty(n, dtype=float)

    for i in range(n):
        df_loo = pd.concat([df.iloc[:i], df.iloc[i + 1 :]])
        theta_loo[i] = statistic(df_loo)

    theta_bar = float(np.mean(theta_loo))
    jackknife_var = ((n - 1) / n) * np.sum((theta_loo - theta_bar) ** 2)
    jackknife_se = float(np.sqrt(jackknife_var))
    jackknife_bias = (n - 1) * (theta_bar - theta_full)

    return {
        "estimate": float(theta_full),
        "se": jackknife_se,
        "bias": float(jackknife_bias),
    }


# ---------------------------------------------------------------------------
# Design weights
# ---------------------------------------------------------------------------


def compute_design_weights(
    df: pd.DataFrame,
    strata_col: str,
    population_sizes: dict[str, int],
) -> pd.Series:
    """Compute inverse-probability design weights for a stratified sample.

    For each stratum *h*, the design weight is:

    .. math::

        w_{hi} = \\frac{N_h}{n_h}

    where :math:`N_h` is the population size and :math:`n_h` is the sample
    size in stratum *h*.

    Parameters
    ----------
    df : pandas.DataFrame
        Sample data.
    strata_col : str
        Column defining strata.
    population_sizes : dict[str, int]
        Mapping of stratum value to population size.

    Returns
    -------
    pandas.Series
        Design weights aligned with the DataFrame index.

    Raises
    ------
    KeyError
        If a stratum in *df* is not found in *population_sizes*.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"stratum": ["A"]*10 + ["B"]*20})
    >>> w = compute_design_weights(df, "stratum", {"A": 1000, "B": 2000})
    >>> float(w[df["stratum"] == "A"].iloc[0])
    100.0

    References
    ----------
    Kish, L. (1965). *Survey Sampling*, Chapter 2. Wiley.
    """
    weights = pd.Series(np.nan, index=df.index, dtype=float, name="design_weight")
    sample_sizes = df[strata_col].value_counts()

    for stratum_val, n_h in sample_sizes.items():
        stratum_key = str(stratum_val)
        if stratum_key not in population_sizes:
            # Try the raw value as well
            if stratum_val not in population_sizes:
                raise KeyError(
                    f"Stratum '{stratum_val}' not found in population_sizes. "
                    f"Available strata: {list(population_sizes.keys())}"
                )
            N_h = population_sizes[stratum_val]
        else:
            N_h = population_sizes[stratum_key]

        mask = df[strata_col] == stratum_val
        weights[mask] = N_h / n_h

    return weights


# ---------------------------------------------------------------------------
# Effective sample size and design effect
# ---------------------------------------------------------------------------


def effective_sample_size(weights: np.ndarray | pd.Series) -> float:
    """Compute the Kish effective sample size.

    .. math::

        n_{\\text{eff}} = \\frac{\\left(\\sum_i w_i\\right)^2}{\\sum_i w_i^2}

    Parameters
    ----------
    weights : numpy.ndarray or pandas.Series
        Sampling or analytic weights.

    Returns
    -------
    float
        Effective sample size.

    Examples
    --------
    >>> import numpy as np
    >>> effective_sample_size(np.ones(100))
    100.0
    >>> ess = effective_sample_size(np.array([1.0, 1.0, 10.0]))
    >>> ess < 3.0
    True

    References
    ----------
    Kish, L. (1965). *Survey Sampling*, p. 162. Wiley.
    """
    w = np.asarray(weights, dtype=float)
    if len(w) == 0:
        return 0.0
    return float((w.sum() ** 2) / np.sum(w**2))


def design_effect(weights: np.ndarray | pd.Series) -> float:
    """Approximate design effect (DEFF) from weights.

    The Kish approximation to the design effect is:

    .. math::

        \\text{DEFF} \\approx \\frac{n}{n_{\\text{eff}}}
        = \\frac{n \\sum_i w_i^2}{\\left(\\sum_i w_i\\right)^2}

    A DEFF of 1.0 indicates no inflation (equal weights); values > 1
    indicate variance inflation from unequal weighting.

    Parameters
    ----------
    weights : numpy.ndarray or pandas.Series
        Sampling or analytic weights.

    Returns
    -------
    float
        Approximate design effect.

    Examples
    --------
    >>> import numpy as np
    >>> design_effect(np.ones(100))
    1.0

    References
    ----------
    Kish, L. (1965). *Survey Sampling*, p. 162. Wiley.
    """
    w = np.asarray(weights, dtype=float)
    n = len(w)
    if n == 0:
        return 1.0
    ess = effective_sample_size(w)
    if ess == 0:
        return float("inf")
    return float(n / ess)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _warn_missing(df: pd.DataFrame, col: str, func_name: str) -> None:
    """Emit a warning if *col* contains missing values."""
    n_missing = int(df[col].isna().sum())
    if n_missing > 0:
        warnings.warn(
            f"{func_name}: column '{col}' has {n_missing} missing values; rows with missing '{col}' will be dropped.",
            stacklevel=3,
        )
