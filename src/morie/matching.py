"""
Matching methods for causal inference in observational studies.

Implements propensity score matching (nearest-neighbour, caliper, with/without
replacement), exact matching, coarsened exact matching (CEM), Mahalanobis
distance matching, optimal matching, genetic matching, entropy balancing,
cardinality matching, subclassification, full matching, balance diagnostics,
treatment effect estimation, Abadie-Imbens standard errors, Rosenbaum
bounds, doubly-robust estimation, and multi-treatment/longitudinal matching.

References
----------
Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity
score in observational studies for causal effects. *Biometrika*, 70(1),
41--55. https://doi.org/10.1093/biomet/70.1.41

Stuart, E. A. (2010). Matching methods for causal inference: A review
and a look forward. *Statistical Science*, 25(1), 1--21.
https://doi.org/10.1214/09-STS313

Iacus, S. M., King, G., & Porro, G. (2012). Causal inference without
balance checking: Coarsened exact matching. *Political Analysis*, 20(1),
1--24. https://doi.org/10.1093/pan/mpr013

Abadie, A., & Imbens, G. W. (2006). Large sample properties of matching
estimators for average treatment effects. *Econometrica*, 74(1), 235--267.
https://doi.org/10.1111/j.1468-0262.2006.00655.x

Hainmueller, J. (2012). Entropy balancing for causal effects: A
multivariate reweighting method to produce balanced samples in
observational studies. *Political Analysis*, 20(1), 25--46.
https://doi.org/10.1093/pan/mpr025
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class MatchResult:
    """Container for matching output.

    Attributes
    ----------
    matched_data : pd.DataFrame
        Matched dataset with match indicators.
    n_treated : int
        Number of treated units.
    n_matched_control : int
        Number of matched control units.
    match_pairs : pd.DataFrame
        Mapping from treated to matched control indices.
    method : str
        Matching method used.
    details : dict
        Additional diagnostics (e.g., caliper, replacement).
    """

    matched_data: pd.DataFrame
    n_treated: int
    n_matched_control: int
    match_pairs: pd.DataFrame
    method: str = "nearest_neighbor"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class BalanceResult:
    """Container for balance diagnostics.

    Attributes
    ----------
    balance_table : pd.DataFrame
        Table with standardised mean differences, variance ratios, etc.
    overall_balance : float
        Mean absolute SMD across covariates.
    max_smd : float
        Maximum absolute SMD.
    balanced : bool
        Whether all SMDs are below the threshold (default 0.1).
    """

    balance_table: pd.DataFrame
    overall_balance: float
    max_smd: float
    balanced: bool


@dataclass
class TreatmentEffectResult:
    """Container for treatment effect estimates from matched samples.

    Attributes
    ----------
    estimand : str
        ATT, ATE, or ATC.
    estimate : float
        Point estimate.
    std_error : float
        Standard error.
    ci_lower : float
        Lower CI bound.
    ci_upper : float
        Upper CI bound.
    p_value : float
        Two-sided p-value.
    n_obs : int
        Effective sample size.
    details : dict
        Extra information.
    """

    estimand: str
    estimate: float
    std_error: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_obs: int
    details: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """One-row summary."""
        return pd.DataFrame(
            [
                {
                    "estimand": self.estimand,
                    "estimate": self.estimate,
                    "std_error": self.std_error,
                    "ci_lower": self.ci_lower,
                    "ci_upper": self.ci_upper,
                    "p_value": self.p_value,
                    "n": self.n_obs,
                }
            ]
        )


# ---------------------------------------------------------------------------
# Propensity score estimation
# ---------------------------------------------------------------------------


def estimate_propensity_score(
    data: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    *,
    model: str = "logistic",
    max_iter: int = 1000,
) -> pd.Series:
    """Estimate propensity scores via logistic regression.

    Parameters
    ----------
    data : pd.DataFrame
    treatment : str
        Binary treatment column (0/1).
    covariates : list of str
        Covariates for the propensity model.
    model : str
        ``"logistic"`` (default) or ``"gbm"`` for gradient boosting.
    max_iter : int
        Maximum iterations for logistic regression.

    Returns
    -------
    pd.Series
        Propensity scores indexed like *data*.

    References
    ----------
    Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the
    propensity score in observational studies for causal effects.
    *Biometrika*, 70(1), 41--55.
    """
    df = data.dropna(subset=[treatment] + covariates).copy()
    X = df[covariates].values.astype(float)
    y = df[treatment].values.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model == "gbm":
        from sklearn.ensemble import GradientBoostingClassifier

        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    else:
        clf = LogisticRegression(max_iter=max_iter, solver="lbfgs")

    clf.fit(X_scaled, y)
    ps = clf.predict_proba(X_scaled)[:, 1]
    return pd.Series(ps, index=df.index, name="propensity_score")


# ---------------------------------------------------------------------------
# Propensity score trimming and truncation
# ---------------------------------------------------------------------------


def trim_propensity_scores(
    ps: pd.Series,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """Clip propensity scores to ``[lower, upper]``.

    Parameters
    ----------
    ps : pd.Series
        Raw propensity scores.
    lower, upper : float

    Returns
    -------
    pd.Series
    """
    return ps.clip(lower=lower, upper=upper)


def common_support(
    data: pd.DataFrame,
    treatment: str,
    ps_col: str = "propensity_score",
    *,
    method: str = "minmax",
) -> pd.DataFrame:
    """Restrict sample to the region of common support.

    Parameters
    ----------
    data : pd.DataFrame
    treatment : str
    ps_col : str
        Propensity score column name.
    method : str
        ``"minmax"`` (overlap of ranges) or ``"trim"`` (drop extreme
        5 % from each tail).

    Returns
    -------
    pd.DataFrame
        Subset with common support.
    """
    df = data.copy()
    ps_t = df.loc[df[treatment] == 1, ps_col]
    ps_c = df.loc[df[treatment] == 0, ps_col]

    if method == "minmax":
        lower = max(ps_t.min(), ps_c.min())
        upper = min(ps_t.max(), ps_c.max())
    else:
        lower = max(ps_t.quantile(0.05), ps_c.quantile(0.05))
        upper = min(ps_t.quantile(0.95), ps_c.quantile(0.95))

    mask = (df[ps_col] >= lower) & (df[ps_col] <= upper)
    logger.info("Common support: dropped %d observations (%.1f%%)", (~mask).sum(), 100 * (~mask).mean())
    return df[mask].copy()


# ---------------------------------------------------------------------------
# Nearest-neighbour propensity score matching
# ---------------------------------------------------------------------------


def match_nearest_neighbor(
    data: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    *,
    n_neighbors: int = 1,
    caliper: float | None = None,
    replace: bool = False,
    ps: pd.Series | None = None,
    alpha: float = 0.05,
) -> MatchResult:
    """Nearest-neighbour matching on propensity score.

    For each treated unit, finds the *n_neighbors* closest control units
    by propensity score distance.

    Parameters
    ----------
    data : pd.DataFrame
    treatment : str
    covariates : list of str
    n_neighbors : int
        Number of matches per treated unit.
    caliper : float, optional
        Maximum propensity score distance for a valid match (in SD of
        the logit of the propensity score).
    replace : bool
        Whether controls can be matched to multiple treated units.
    ps : pd.Series, optional
        Pre-computed propensity scores.  If ``None``, estimated
        internally.
    alpha : float
        Significance level for balance checks.

    Returns
    -------
    MatchResult
    """
    df = data.dropna(subset=[treatment] + covariates).copy()
    if ps is None:
        ps = estimate_propensity_score(df, treatment, covariates)
    df["_ps"] = ps.reindex(df.index)

    treated_idx = df.index[df[treatment] == 1].tolist()
    control_idx = df.index[df[treatment] == 0].tolist()
    ps_treated = df.loc[treated_idx, "_ps"].values
    ps_control = df.loc[control_idx, "_ps"].values

    # Convert to logit for distance calculation
    eps = 1e-6
    logit_t = np.log(np.clip(ps_treated, eps, 1 - eps) / (1 - np.clip(ps_treated, eps, 1 - eps)))
    logit_c = np.log(np.clip(ps_control, eps, 1 - eps) / (1 - np.clip(ps_control, eps, 1 - eps)))

    if caliper is not None:
        caliper_val = caliper * np.std(np.concatenate([logit_t, logit_c]))
    else:
        caliper_val = np.inf

    nn = NearestNeighbors(n_neighbors=min(n_neighbors * 5, len(control_idx)), metric="euclidean")
    nn.fit(logit_c.reshape(-1, 1))
    distances, indices = nn.kneighbors(logit_t.reshape(-1, 1))

    match_records = []
    used_controls = set()

    for i, t_idx in enumerate(treated_idx):
        matched = []
        for j in range(indices.shape[1]):
            c_pos = indices[i, j]
            c_idx = control_idx[c_pos]
            dist = distances[i, j]

            if dist > caliper_val:
                break
            if not replace and c_idx in used_controls:
                continue

            matched.append(c_idx)
            used_controls.add(c_idx)

            if len(matched) >= n_neighbors:
                break

        for c_idx in matched:
            match_records.append(
                {
                    "treated_idx": t_idx,
                    "control_idx": c_idx,
                    "distance": float(abs(df.loc[t_idx, "_ps"] - df.loc[c_idx, "_ps"])),
                }
            )

    match_df = pd.DataFrame(match_records)
    matched_control_ids = set(match_df["control_idx"]) if len(match_df) > 0 else set()
    matched_treated_ids = set(match_df["treated_idx"]) if len(match_df) > 0 else set()
    all_matched_ids = list(matched_treated_ids | matched_control_ids)
    matched_data = df.loc[df.index.isin(all_matched_ids)].copy()

    return MatchResult(
        matched_data=matched_data,
        n_treated=len(matched_treated_ids),
        n_matched_control=len(matched_control_ids),
        match_pairs=match_df,
        method="nearest_neighbor",
        details={
            "caliper": caliper_val,
            "replace": replace,
            "n_neighbors": n_neighbors,
            "n_unmatched": len(treated_idx) - len(matched_treated_ids),
        },
    )


# ---------------------------------------------------------------------------
# Exact matching
# ---------------------------------------------------------------------------


def match_exact(
    data: pd.DataFrame,
    treatment: str,
    exact_vars: list[str],
) -> MatchResult:
    """Exact matching on discrete covariates.

    Matches treated and control units that share identical values on
    all *exact_vars*.  Unmatched units are dropped.

    Parameters
    ----------
    data : pd.DataFrame
    treatment : str
    exact_vars : list of str

    Returns
    -------
    MatchResult
    """
    df = data.dropna(subset=[treatment] + exact_vars).copy()
    df["_stratum"] = df[exact_vars].astype(str).agg("|".join, axis=1)

    records = []
    matched_ids = []

    for stratum, grp in df.groupby("_stratum"):
        t_ids = grp.index[grp[treatment] == 1].tolist()
        c_ids = grp.index[grp[treatment] == 0].tolist()
        if len(t_ids) == 0 or len(c_ids) == 0:
            continue
        for t_id in t_ids:
            for c_id in c_ids:
                records.append({"treated_idx": t_id, "control_idx": c_id, "distance": 0.0})
            matched_ids.append(t_id)
        matched_ids.extend(c_ids)

    match_df = pd.DataFrame(records) if records else pd.DataFrame(columns=["treated_idx", "control_idx", "distance"])
    matched_data = df.loc[df.index.isin(set(matched_ids))].drop(columns=["_stratum"])

    return MatchResult(
        matched_data=matched_data,
        n_treated=len(set(r["treated_idx"] for r in records)) if records else 0,
        n_matched_control=len(set(r["control_idx"] for r in records)) if records else 0,
        match_pairs=match_df,
        method="exact",
    )


# ---------------------------------------------------------------------------
# Coarsened exact matching (CEM)
# ---------------------------------------------------------------------------


def match_cem(
    data: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    *,
    n_bins: Union[int, dict[str, int]] = 5,
) -> MatchResult:
    """Coarsened Exact Matching (Iacus, King & Porro, 2012).

    Coarsens continuous covariates into bins and performs exact matching
    on the coarsened values.  Returns the original (uncoarsened) data
    with CEM strata weights.

    Parameters
    ----------
    data : pd.DataFrame
    treatment : str
    covariates : list of str
    n_bins : int or dict
        Number of bins for each covariate.  A single int applies to all.

    Returns
    -------
    MatchResult
        ``matched_data`` includes a ``_cem_weight`` column.

    References
    ----------
    Iacus, S. M., King, G., & Porro, G. (2012). Causal inference without
    balance checking: Coarsened exact matching. *Political Analysis*,
    20(1), 1--24.
    """
    df = data.dropna(subset=[treatment] + covariates).copy()

    if isinstance(n_bins, int):
        bins_map = {c: n_bins for c in covariates}
    else:
        bins_map = n_bins

    for c in covariates:
        nb = bins_map.get(c, 5)
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > nb:
            df[f"_cem_{c}"] = pd.qcut(df[c], nb, labels=False, duplicates="drop")
        else:
            df[f"_cem_{c}"] = df[c].astype(str)

    cem_cols = [f"_cem_{c}" for c in covariates]
    df["_cem_stratum"] = df[cem_cols].astype(str).agg("|".join, axis=1)

    # Keep only strata with both treated and control
    valid_strata = set()
    for s, grp in df.groupby("_cem_stratum"):
        if grp[treatment].nunique() == 2:
            valid_strata.add(s)

    df_matched = df[df["_cem_stratum"].isin(valid_strata)].copy()

    # Compute CEM weights
    weights = np.ones(len(df_matched))
    for s, grp in df_matched.groupby("_cem_stratum"):
        n_t = (grp[treatment] == 1).sum()
        n_c = (grp[treatment] == 0).sum()
        if n_t == 0 or n_c == 0:
            continue
        # Weight controls to have same total as treated within stratum
        t_mask = df_matched.index.isin(grp.index[grp[treatment] == 1])
        c_mask = df_matched.index.isin(grp.index[grp[treatment] == 0])
        weights[np.where(c_mask[df_matched.index].values)[0] if hasattr(c_mask, "values") else c_mask] = n_t / n_c

    # Simpler weight assignment
    df_matched["_cem_weight"] = 1.0
    for s in valid_strata:
        mask_s = df_matched["_cem_stratum"] == s
        n_t = (df_matched.loc[mask_s, treatment] == 1).sum()
        n_c = (df_matched.loc[mask_s, treatment] == 0).sum()
        if n_c > 0:
            df_matched.loc[mask_s & (df_matched[treatment] == 0), "_cem_weight"] = n_t / n_c

    # Build match pairs
    records = []
    for s in valid_strata:
        grp = df_matched[df_matched["_cem_stratum"] == s]
        t_ids = grp.index[grp[treatment] == 1].tolist()
        c_ids = grp.index[grp[treatment] == 0].tolist()
        for t_id in t_ids:
            for c_id in c_ids:
                records.append({"treated_idx": t_id, "control_idx": c_id, "distance": 0.0})

    match_df = pd.DataFrame(records) if records else pd.DataFrame(columns=["treated_idx", "control_idx", "distance"])

    # Drop CEM helper columns
    drop_cols = cem_cols + ["_cem_stratum"]
    df_matched = df_matched.drop(columns=drop_cols, errors="ignore")

    return MatchResult(
        matched_data=df_matched,
        n_treated=int((df_matched[treatment] == 1).sum()),
        n_matched_control=int((df_matched[treatment] == 0).sum()),
        match_pairs=match_df,
        method="cem",
        details={"n_strata": len(valid_strata)},
    )


# ---------------------------------------------------------------------------
# Mahalanobis distance matching
# ---------------------------------------------------------------------------


def match_mahalanobis(
    data: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    *,
    n_neighbors: int = 1,
    caliper: float | None = None,
    replace: bool = False,
    exact: list[str] | None = None,
) -> MatchResult:
    """Mahalanobis distance matching.

    Matches on the Mahalanobis distance of covariates rather than
    propensity score.  Optionally combined with exact matching on
    discrete variables.

    Parameters
    ----------
    data : pd.DataFrame
    treatment : str
    covariates : list of str
        Continuous covariates for distance calculation.
    n_neighbors : int
    caliper : float, optional
    replace : bool
    exact : list of str, optional
        Variables to match exactly before distance matching.

    Returns
    -------
    MatchResult
    """
    df = data.dropna(subset=[treatment] + covariates).copy()
    X = df[covariates].values.astype(float)

    # Compute Mahalanobis distance covariance
    cov_matrix = np.cov(X, rowvar=False)
    try:
        cov_inv = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_matrix)

    treated_mask = df[treatment] == 1
    control_mask = df[treatment] == 0
    treated_idx = df.index[treated_mask].tolist()
    control_idx = df.index[control_mask].tolist()
    X_t = X[treated_mask.values]
    X_c = X[control_mask.values]

    # Compute pairwise Mahalanobis distances
    dist_matrix = cdist(X_t, X_c, metric="mahalanobis", VI=cov_inv)

    match_records = []
    used_controls = set()

    for i, t_idx in enumerate(treated_idx):
        if exact:
            # Filter controls that match exactly
            valid_j = []
            for j, c_idx in enumerate(control_idx):
                if all(df.loc[t_idx, v] == df.loc[c_idx, v] for v in exact):
                    valid_j.append(j)
            if not valid_j:
                continue
            dists = [(dist_matrix[i, j], j) for j in valid_j]
        else:
            dists = [(dist_matrix[i, j], j) for j in range(len(control_idx))]

        dists.sort(key=lambda x: x[0])

        matched = []
        for d, j in dists:
            c_idx = control_idx[j]
            if caliper is not None and d > caliper:
                break
            if not replace and c_idx in used_controls:
                continue
            matched.append((c_idx, d))
            used_controls.add(c_idx)
            if len(matched) >= n_neighbors:
                break

        for c_idx, d in matched:
            match_records.append({"treated_idx": t_idx, "control_idx": c_idx, "distance": float(d)})

    match_df = (
        pd.DataFrame(match_records)
        if match_records
        else pd.DataFrame(columns=["treated_idx", "control_idx", "distance"])
    )
    all_ids = set()
    for r in match_records:
        all_ids.add(r["treated_idx"])
        all_ids.add(r["control_idx"])
    matched_data = df.loc[df.index.isin(all_ids)].copy()

    return MatchResult(
        matched_data=matched_data,
        n_treated=len(set(r["treated_idx"] for r in match_records)) if match_records else 0,
        n_matched_control=len(set(r["control_idx"] for r in match_records)) if match_records else 0,
        match_pairs=match_df,
        method="mahalanobis",
        details={"caliper": caliper, "replace": replace, "exact_vars": exact},
    )


# ---------------------------------------------------------------------------
# Optimal matching (pair and full)
# ---------------------------------------------------------------------------


def match_optimal_pair(
    data: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    *,
    distance: str = "propensity",
    ps: pd.Series | None = None,
) -> MatchResult:
    """Optimal pair matching minimising total distance.

    Uses a greedy approximation to the assignment problem (Hungarian
    algorithm is O(n^3); we use sorted nearest-neighbor for scalability).

    Parameters
    ----------
    data : pd.DataFrame
    treatment : str
    covariates : list of str
    distance : str
        ``"propensity"`` or ``"mahalanobis"``.
    ps : pd.Series, optional

    Returns
    -------
    MatchResult
    """
    df = data.dropna(subset=[treatment] + covariates).copy()

    if distance == "propensity":
        if ps is None:
            ps = estimate_propensity_score(df, treatment, covariates)
        ps_vals = ps.reindex(df.index).values.astype(float)
        X_t = ps_vals[df[treatment].values == 1].reshape(-1, 1)
        X_c = ps_vals[df[treatment].values == 0].reshape(-1, 1)
        dist_matrix = cdist(X_t, X_c, metric="euclidean")
    else:
        X = df[covariates].values.astype(float)
        cov_inv = np.linalg.pinv(np.cov(X, rowvar=False))
        X_t = X[df[treatment].values == 1]
        X_c = X[df[treatment].values == 0]
        dist_matrix = cdist(X_t, X_c, metric="mahalanobis", VI=cov_inv)

    treated_idx = df.index[df[treatment] == 1].tolist()
    control_idx = df.index[df[treatment] == 0].tolist()

    n_t = len(treated_idx)
    n_c = len(control_idx)
    n_pairs = min(n_t, n_c)

    # Greedy optimal: sort all pairwise distances and greedily assign
    flat_dists = []
    for i in range(n_t):
        for j in range(n_c):
            flat_dists.append((dist_matrix[i, j], i, j))
    flat_dists.sort(key=lambda x: x[0])

    used_t = set()
    used_c = set()
    match_records = []

    for d, i, j in flat_dists:
        if i in used_t or j in used_c:
            continue
        match_records.append(
            {
                "treated_idx": treated_idx[i],
                "control_idx": control_idx[j],
                "distance": float(d),
            }
        )
        used_t.add(i)
        used_c.add(j)
        if len(match_records) >= n_pairs:
            break

    match_df = pd.DataFrame(match_records)
    all_ids = set()
    for r in match_records:
        all_ids.add(r["treated_idx"])
        all_ids.add(r["control_idx"])
    matched_data = df.loc[df.index.isin(all_ids)].copy()

    return MatchResult(
        matched_data=matched_data,
        n_treated=len(used_t),
        n_matched_control=len(used_c),
        match_pairs=match_df,
        method="optimal_pair",
    )


# ---------------------------------------------------------------------------
# Full matching (Hansen, 2004)
# ---------------------------------------------------------------------------


def match_full(
    data: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    *,
    ps: pd.Series | None = None,
    n_subclasses: int = 10,
) -> MatchResult:
    """Full matching via propensity score subclassification.

    Every unit is placed into a subclass containing at least one treated
    and one control unit.  Approximated here via quantile-based
    stratification of the propensity score.

    Parameters
    ----------
    data : pd.DataFrame
    treatment : str
    covariates : list of str
    ps : pd.Series, optional
    n_subclasses : int
        Number of propensity score strata.

    Returns
    -------
    MatchResult

    References
    ----------
    Hansen, B. B. (2004). Full matching in an observational study of
    coaching for the SAT. *Journal of the American Statistical
    Association*, 99(467), 609--618.
    """
    df = data.dropna(subset=[treatment] + covariates).copy()
    if ps is None:
        ps = estimate_propensity_score(df, treatment, covariates)
    df["_ps"] = ps.reindex(df.index)

    # Stratify
    df["_subclass"] = pd.qcut(df["_ps"], n_subclasses, labels=False, duplicates="drop")

    # Drop strata without both treated and control
    valid = df.groupby("_subclass")[treatment].apply(lambda x: x.nunique() == 2)
    valid_strata = valid[valid].index.tolist()
    df_matched = df[df["_subclass"].isin(valid_strata)].copy()

    # Compute subclass weights
    df_matched["_full_weight"] = 1.0
    for s in valid_strata:
        mask_s = df_matched["_subclass"] == s
        n_t = (df_matched.loc[mask_s, treatment] == 1).sum()
        n_c = (df_matched.loc[mask_s, treatment] == 0).sum()
        n_total = n_t + n_c
        if n_c > 0:
            df_matched.loc[mask_s & (df_matched[treatment] == 0), "_full_weight"] = n_t / n_c
        if n_t > 0:
            df_matched.loc[mask_s & (df_matched[treatment] == 1), "_full_weight"] = 1.0

    # Match pairs (each treated to all controls in its subclass)
    records = []
    for s in valid_strata:
        grp = df_matched[df_matched["_subclass"] == s]
        t_ids = grp.index[grp[treatment] == 1].tolist()
        c_ids = grp.index[grp[treatment] == 0].tolist()
        for t_id in t_ids:
            for c_id in c_ids:
                records.append(
                    {
                        "treated_idx": t_id,
                        "control_idx": c_id,
                        "distance": abs(df_matched.loc[t_id, "_ps"] - df_matched.loc[c_id, "_ps"]),
                    }
                )

    match_df = pd.DataFrame(records) if records else pd.DataFrame(columns=["treated_idx", "control_idx", "distance"])
    df_matched = df_matched.drop(columns=["_subclass", "_ps"], errors="ignore")

    return MatchResult(
        matched_data=df_matched,
        n_treated=int((df_matched[treatment] == 1).sum()),
        n_matched_control=int((df_matched[treatment] == 0).sum()),
        match_pairs=match_df,
        method="full_matching",
        details={"n_subclasses": len(valid_strata)},
    )


# ---------------------------------------------------------------------------
# Subclassification on propensity score
# ---------------------------------------------------------------------------


def subclassify(
    data: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    *,
    ps: pd.Series | None = None,
    n_strata: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subclassification (stratification) on the propensity score.

    Parameters
    ----------
    data : pd.DataFrame
    treatment : str
    covariates : list of str
    ps : pd.Series, optional
    n_strata : int

    Returns
    -------
    data_with_strata : pd.DataFrame
        Input data with a ``_stratum`` column appended.
    stratum_effects : pd.DataFrame
        Within-stratum treatment effect estimates.
    """
    df = data.dropna(subset=[treatment] + covariates).copy()
    if ps is None:
        ps = estimate_propensity_score(df, treatment, covariates)
    df["_ps"] = ps.reindex(df.index)
    df["_stratum"] = pd.qcut(df["_ps"], n_strata, labels=False, duplicates="drop")

    records = []
    for s, grp in df.groupby("_stratum"):
        y_t = grp.loc[grp[treatment] == 1]
        y_c = grp.loc[grp[treatment] == 0]
        if len(y_t) == 0 or len(y_c) == 0:
            continue
        records.append(
            {
                "stratum": s,
                "n_treated": len(y_t),
                "n_control": len(y_c),
                "ps_range_low": float(grp["_ps"].min()),
                "ps_range_high": float(grp["_ps"].max()),
            }
        )

    return df, pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Entropy balancing (Hainmueller, 2012)
# ---------------------------------------------------------------------------


def entropy_balance(
    data: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    *,
    max_moment: int = 1,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> pd.Series:
    r"""Entropy balancing weights for the control group.

    Finds weights :math:`w_i` for control units that minimise the
    Kullback-Leibler divergence from uniform weights subject to moment
    balance constraints:

    .. math::

        \sum_{i: D_i=0} w_i\, x_i^k = \bar x^k_{D=1}
        \quad \text{for } k = 1, \ldots, \text{max\_moment}

    Parameters
    ----------
    data : pd.DataFrame
    treatment : str
    covariates : list of str
    max_moment : int
        Balance on mean (1), mean + variance (2), or up to 3rd moment.
    max_iter : int
    tol : float

    Returns
    -------
    pd.Series
        Entropy balancing weights indexed like *data*.  Treated units
        receive weight 1; control units receive the balancing weights.

    References
    ----------
    Hainmueller, J. (2012). Entropy balancing for causal effects.
    *Political Analysis*, 20(1), 25--46.
    """
    df = data.dropna(subset=[treatment] + covariates).copy()
    t_mask = df[treatment] == 1
    c_mask = df[treatment] == 0

    X_t = df.loc[t_mask, covariates].values.astype(float)
    X_c = df.loc[c_mask, covariates].values.astype(float)
    n_c = X_c.shape[0]

    # Build moment constraints
    targets = []
    C_matrix = []

    for m in range(1, max_moment + 1):
        targets.append(np.mean(X_t**m, axis=0))
        C_matrix.append(X_c**m)

    targets = np.concatenate(targets)
    C = np.hstack(C_matrix)  # (n_c, n_constraints)

    # Solve dual problem via Newton's method
    n_constraints = len(targets)
    lam = np.zeros(n_constraints)

    for _ in range(max_iter):
        # Weights: w_i = exp(C_i' lambda) / sum
        logits = C @ lam
        logits -= logits.max()  # numerical stability
        w_raw = np.exp(logits)
        w = w_raw / w_raw.sum()

        # Gradient: C'w - targets
        g = C.T @ w - targets
        if np.max(np.abs(g)) < tol:
            break

        # Hessian: C' diag(w) C - (C'w)(C'w)'
        H = C.T @ np.diag(w) @ C - np.outer(C.T @ w, C.T @ w)
        try:
            dlam = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            dlam = np.linalg.lstsq(H, -g, rcond=None)[0]

        lam += dlam

    # Final weights
    logits = C @ lam
    logits -= logits.max()
    w_raw = np.exp(logits)
    w_final = w_raw / w_raw.sum() * n_c  # normalise to sum to n_c

    weights = pd.Series(1.0, index=df.index, name="ebal_weight")
    weights.loc[c_mask] = w_final

    return weights


# ---------------------------------------------------------------------------
# Genetic matching
# ---------------------------------------------------------------------------


def match_genetic(
    data: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    *,
    n_neighbors: int = 1,
    pop_size: int = 50,
    n_generations: int = 20,
    seed: int = 42,
) -> MatchResult:
    """Genetic matching (Diamond & Sekhon, 2013).

    Uses a genetic algorithm to find the optimal weight matrix for
    Mahalanobis distance matching that maximises covariate balance.

    Parameters
    ----------
    data : pd.DataFrame
    treatment : str
    covariates : list of str
    n_neighbors : int
    pop_size : int
        Genetic algorithm population size.
    n_generations : int
    seed : int

    Returns
    -------
    MatchResult

    References
    ----------
    Diamond, A., & Sekhon, J. S. (2013). Genetic matching for estimating
    causal effects. *Review of Economics and Statistics*, 95(3), 932--945.
    """
    rng = np.random.default_rng(seed)
    df = data.dropna(subset=[treatment] + covariates).copy()
    X = df[covariates].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    p = X_scaled.shape[1]

    treated_mask = df[treatment].values == 1
    X_t = X_scaled[treated_mask]
    X_c = X_scaled[~treated_mask]
    treated_idx = df.index[treated_mask].tolist()
    control_idx = df.index[~treated_mask].tolist()

    def _evaluate_weights(w):
        """Compute balance metric for given distance weights."""
        W = np.diag(np.abs(w))
        X_t_w = X_t @ W
        X_c_w = X_c @ W
        dist = cdist(X_t_w, X_c_w, metric="euclidean")
        # Greedy 1:1 matching
        used = set()
        total_smd = 0.0
        matched_c_indices = []
        for i in range(len(X_t)):
            best_j = None
            best_d = np.inf
            for j in range(len(X_c)):
                if j not in used and dist[i, j] < best_d:
                    best_d = dist[i, j]
                    best_j = j
            if best_j is not None:
                used.add(best_j)
                matched_c_indices.append(best_j)

        if len(matched_c_indices) == 0:
            return 1e6

        X_c_matched = X_c[matched_c_indices]
        smds = np.abs(X_t[: len(matched_c_indices)].mean(axis=0) - X_c_matched.mean(axis=0))
        pooled_sd = np.sqrt((X_t.var(axis=0) + X_c_matched.var(axis=0)) / 2 + 1e-10)
        return float(np.max(smds / pooled_sd))

    # Genetic algorithm
    population = rng.uniform(0.1, 2.0, size=(pop_size, p))
    best_w = np.ones(p)
    best_score = _evaluate_weights(best_w)

    for gen in range(n_generations):
        scores = np.array([_evaluate_weights(ind) for ind in population])
        sorted_idx = np.argsort(scores)
        if scores[sorted_idx[0]] < best_score:
            best_score = scores[sorted_idx[0]]
            best_w = population[sorted_idx[0]].copy()

        # Selection: keep top half
        parents = population[sorted_idx[: pop_size // 2]]

        # Crossover and mutation
        children = []
        for _ in range(pop_size - len(parents)):
            p1, p2 = parents[rng.choice(len(parents), 2, replace=False)]
            mask = rng.random(p) > 0.5
            child = np.where(mask, p1, p2)
            # Mutation
            if rng.random() < 0.3:
                mut_idx = rng.choice(p)
                child[mut_idx] *= rng.uniform(0.5, 1.5)
            children.append(child)

        population = np.vstack([parents, children])

    # Final matching with best weights
    W = np.diag(np.abs(best_w))
    X_t_w = X_t @ W
    X_c_w = X_c @ W
    dist = cdist(X_t_w, X_c_w, metric="euclidean")

    match_records = []
    used = set()
    for i in range(len(X_t)):
        sorted_j = np.argsort(dist[i])
        matched_count = 0
        for j in sorted_j:
            if j in used:
                continue
            match_records.append(
                {
                    "treated_idx": treated_idx[i],
                    "control_idx": control_idx[j],
                    "distance": float(dist[i, j]),
                }
            )
            used.add(j)
            matched_count += 1
            if matched_count >= n_neighbors:
                break

    match_df = (
        pd.DataFrame(match_records)
        if match_records
        else pd.DataFrame(columns=["treated_idx", "control_idx", "distance"])
    )
    all_ids = set(r["treated_idx"] for r in match_records) | set(r["control_idx"] for r in match_records)
    matched_data = df.loc[df.index.isin(all_ids)].copy()

    return MatchResult(
        matched_data=matched_data,
        n_treated=len(set(r["treated_idx"] for r in match_records)),
        n_matched_control=len(set(r["control_idx"] for r in match_records)),
        match_pairs=match_df,
        method="genetic",
        details={"best_weights": best_w.tolist(), "best_balance": best_score},
    )


# ---------------------------------------------------------------------------
# Variable ratio matching
# ---------------------------------------------------------------------------


def match_variable_ratio(
    data: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    *,
    min_ratio: int = 1,
    max_ratio: int = 5,
    caliper: float = 0.2,
    ps: pd.Series | None = None,
) -> MatchResult:
    """Variable-ratio matching on propensity score.

    Each treated unit is matched to between *min_ratio* and *max_ratio*
    controls within the caliper.

    Parameters
    ----------
    data : pd.DataFrame
    treatment : str
    covariates : list of str
    min_ratio, max_ratio : int
    caliper : float
    ps : pd.Series, optional

    Returns
    -------
    MatchResult
    """
    df = data.dropna(subset=[treatment] + covariates).copy()
    if ps is None:
        ps = estimate_propensity_score(df, treatment, covariates)
    df["_ps"] = ps.reindex(df.index)

    treated_idx = df.index[df[treatment] == 1].tolist()
    control_idx = df.index[df[treatment] == 0].tolist()

    ps_sd = df["_ps"].std()
    caliper_val = caliper * ps_sd

    match_records = []
    for t_id in treated_idx:
        ps_t = df.loc[t_id, "_ps"]
        dists = []
        for c_id in control_idx:
            d = abs(ps_t - df.loc[c_id, "_ps"])
            if d <= caliper_val:
                dists.append((d, c_id))
        dists.sort()

        n_match = min(max(len(dists), min_ratio), max_ratio)
        n_match = min(n_match, len(dists))
        for i in range(n_match):
            match_records.append(
                {
                    "treated_idx": t_id,
                    "control_idx": dists[i][1],
                    "distance": float(dists[i][0]),
                }
            )

    match_df = (
        pd.DataFrame(match_records)
        if match_records
        else pd.DataFrame(columns=["treated_idx", "control_idx", "distance"])
    )
    all_ids = set()
    for r in match_records:
        all_ids.add(r["treated_idx"])
        all_ids.add(r["control_idx"])
    matched_data = df.loc[df.index.isin(all_ids)].copy()

    return MatchResult(
        matched_data=matched_data,
        n_treated=len(set(r["treated_idx"] for r in match_records)),
        n_matched_control=len(set(r["control_idx"] for r in match_records)),
        match_pairs=match_df,
        method="variable_ratio",
        details={"caliper": caliper_val, "min_ratio": min_ratio, "max_ratio": max_ratio},
    )


# ---------------------------------------------------------------------------
# Cardinality matching
# ---------------------------------------------------------------------------


def match_cardinality(
    data: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    *,
    balance_threshold: float = 0.1,
    ps: pd.Series | None = None,
) -> MatchResult:
    """Cardinality matching: maximise sample size subject to balance.

    Finds the largest subset of matched pairs such that the standardised
    mean difference on each covariate is below *balance_threshold*.

    Uses an iterative caliper-tightening heuristic.

    Parameters
    ----------
    data : pd.DataFrame
    treatment : str
    covariates : list of str
    balance_threshold : float
    ps : pd.Series, optional

    Returns
    -------
    MatchResult

    References
    ----------
    Zubizarreta, J. R. (2012). Using mixed integer programming for matching
    in an observational study of kidney failure after surgery. *Journal of
    the American Statistical Association*, 107(500), 1360--1371.
    """
    best_result = None
    best_n = 0

    for caliper in [None, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05]:
        result = match_nearest_neighbor(
            data,
            treatment,
            covariates,
            caliper=caliper,
            replace=False,
            ps=ps,
        )
        if len(result.matched_data) == 0:
            continue

        bal = balance_diagnostics(result.matched_data, treatment, covariates)
        if bal.max_smd <= balance_threshold:
            if result.n_treated + result.n_matched_control > best_n:
                best_n = result.n_treated + result.n_matched_control
                best_result = result
                best_result.method = "cardinality"
                best_result.details["balance_threshold"] = balance_threshold
                # Don't need to go tighter if we have balance
                break

    if best_result is None:
        # Return the loosest caliper result
        best_result = match_nearest_neighbor(data, treatment, covariates, ps=ps)
        best_result.method = "cardinality"
        best_result.details["warning"] = "Balance threshold not achieved."

    return best_result


# ---------------------------------------------------------------------------
# Balance diagnostics
# ---------------------------------------------------------------------------


def balance_diagnostics(
    data: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    *,
    weights: str | None = None,
    threshold: float = 0.1,
) -> BalanceResult:
    """Compute balance diagnostics for matched/weighted samples.

    Reports standardised mean differences (SMD), variance ratios, and
    Kolmogorov-Smirnov statistics for each covariate.

    Parameters
    ----------
    data : pd.DataFrame
    treatment : str
    covariates : list of str
    weights : str, optional
        Column of matching/weighting weights.
    threshold : float
        SMD threshold for ``balanced`` flag (default 0.1).

    Returns
    -------
    BalanceResult
    """
    df = data.dropna(subset=[treatment] + covariates).copy()
    t_mask = df[treatment] == 1
    c_mask = df[treatment] == 0

    records = []
    for cov in covariates:
        t_vals = df.loc[t_mask, cov].astype(float)
        c_vals = df.loc[c_mask, cov].astype(float)

        if weights and weights in df.columns:
            w_t = df.loc[t_mask, weights].values.astype(float)
            w_c = df.loc[c_mask, weights].values.astype(float)
            mean_t = float(np.average(t_vals, weights=w_t))
            mean_c = float(np.average(c_vals, weights=w_c))
            var_t = float(np.average((t_vals - mean_t) ** 2, weights=w_t))
            var_c = float(np.average((c_vals - mean_c) ** 2, weights=w_c))
        else:
            mean_t = float(t_vals.mean())
            mean_c = float(c_vals.mean())
            var_t = float(t_vals.var(ddof=1)) if len(t_vals) > 1 else 0.0
            var_c = float(c_vals.var(ddof=1)) if len(c_vals) > 1 else 0.0

        pooled_sd = np.sqrt((var_t + var_c) / 2)
        smd = (mean_t - mean_c) / pooled_sd if pooled_sd > 0 else 0.0
        var_ratio = var_t / var_c if var_c > 0 else np.nan

        # KS statistic
        ks_stat, ks_p = stats.ks_2samp(t_vals.values, c_vals.values)

        records.append(
            {
                "covariate": cov,
                "mean_treated": mean_t,
                "mean_control": mean_c,
                "smd": smd,
                "abs_smd": abs(smd),
                "variance_ratio": var_ratio,
                "ks_stat": float(ks_stat),
                "ks_p_value": float(ks_p),
            }
        )

    bal_df = pd.DataFrame(records)
    overall = float(bal_df["abs_smd"].mean()) if len(bal_df) > 0 else 0.0
    max_smd = float(bal_df["abs_smd"].max()) if len(bal_df) > 0 else 0.0

    return BalanceResult(
        balance_table=bal_df,
        overall_balance=overall,
        max_smd=max_smd,
        balanced=max_smd <= threshold,
    )


def love_plot_data(
    unmatched_data: pd.DataFrame,
    matched_data: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    *,
    weights_col: str | None = None,
) -> pd.DataFrame:
    """Generate data for a Love plot comparing pre- and post-matching balance.

    Parameters
    ----------
    unmatched_data : pd.DataFrame
    matched_data : pd.DataFrame
    treatment : str
    covariates : list of str
    weights_col : str, optional

    Returns
    -------
    pd.DataFrame
        Columns: ``covariate``, ``smd_before``, ``smd_after``.
    """
    bal_before = balance_diagnostics(unmatched_data, treatment, covariates)
    bal_after = balance_diagnostics(matched_data, treatment, covariates, weights=weights_col)

    before = bal_before.balance_table.set_index("covariate")["smd"]
    after = bal_after.balance_table.set_index("covariate")["smd"]

    result = pd.DataFrame(
        {
            "covariate": covariates,
            "smd_before": [before.get(c, np.nan) for c in covariates],
            "smd_after": [after.get(c, np.nan) for c in covariates],
        }
    )
    result["abs_smd_before"] = result["smd_before"].abs()
    result["abs_smd_after"] = result["smd_after"].abs()
    return result


def balance_table(
    data: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    *,
    weights: str | None = None,
) -> pd.DataFrame:
    """Construct a publication-ready balance table.

    Parameters
    ----------
    data : pd.DataFrame
    treatment : str
    covariates : list of str
    weights : str, optional

    Returns
    -------
    pd.DataFrame
    """
    bal = balance_diagnostics(data, treatment, covariates, weights=weights)
    return bal.balance_table


# ---------------------------------------------------------------------------
# Treatment effect estimation from matched samples
# ---------------------------------------------------------------------------


def estimate_att_matched(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    match_pairs: pd.DataFrame,
    *,
    weights: str | None = None,
    alpha: float = 0.05,
) -> TreatmentEffectResult:
    """Estimate ATT from matched data.

    Parameters
    ----------
    data : pd.DataFrame
    outcome : str
    treatment : str
    match_pairs : pd.DataFrame
        With columns ``treated_idx`` and ``control_idx``.
    weights : str, optional
    alpha : float

    Returns
    -------
    TreatmentEffectResult
    """
    diffs = []
    for _, row in match_pairs.iterrows():
        t_id = row["treated_idx"]
        c_id = row["control_idx"]
        if t_id in data.index and c_id in data.index:
            y_t = float(data.loc[t_id, outcome])
            y_c = float(data.loc[c_id, outcome])
            diffs.append(y_t - y_c)

    if len(diffs) == 0:
        return TreatmentEffectResult(
            estimand="ATT",
            estimate=np.nan,
            std_error=np.nan,
            ci_lower=np.nan,
            ci_upper=np.nan,
            p_value=np.nan,
            n_obs=0,
        )

    diffs = np.array(diffs)
    att = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
    t_val = att / se if se > 0 else 0.0
    p_val = float(2 * stats.norm.sf(abs(t_val)))
    z = stats.norm.ppf(1 - alpha / 2)

    return TreatmentEffectResult(
        estimand="ATT",
        estimate=att,
        std_error=se,
        ci_lower=att - z * se,
        ci_upper=att + z * se,
        p_value=p_val,
        n_obs=len(diffs),
    )


def estimate_ate_matched(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    covariates: list[str],
    *,
    weights: str | None = None,
    alpha: float = 0.05,
) -> TreatmentEffectResult:
    """Estimate ATE from matched/weighted data via weighted mean difference.

    Parameters
    ----------
    data : pd.DataFrame
    outcome, treatment : str
    covariates : list of str
    weights : str, optional
    alpha : float

    Returns
    -------
    TreatmentEffectResult
    """
    df = data.dropna(subset=[outcome, treatment]).copy()
    t_mask = df[treatment] == 1
    c_mask = df[treatment] == 0
    y_t = df.loc[t_mask, outcome].values.astype(float)
    y_c = df.loc[c_mask, outcome].values.astype(float)

    if weights and weights in df.columns:
        w_t = df.loc[t_mask, weights].values.astype(float)
        w_c = df.loc[c_mask, weights].values.astype(float)
        mean_t = float(np.average(y_t, weights=w_t))
        mean_c = float(np.average(y_c, weights=w_c))
        # Weighted variance
        var_t = float(np.average((y_t - mean_t) ** 2, weights=w_t))
        var_c = float(np.average((y_c - mean_c) ** 2, weights=w_c))
        n_eff_t = w_t.sum() ** 2 / (w_t**2).sum()
        n_eff_c = w_c.sum() ** 2 / (w_c**2).sum()
        se = float(np.sqrt(var_t / n_eff_t + var_c / n_eff_c))
    else:
        mean_t = float(y_t.mean())
        mean_c = float(y_c.mean())
        se = float(np.sqrt(y_t.var(ddof=1) / len(y_t) + y_c.var(ddof=1) / len(y_c)))

    ate = mean_t - mean_c
    t_val = ate / se if se > 0 else 0.0
    p_val = float(2 * stats.norm.sf(abs(t_val)))
    z = stats.norm.ppf(1 - alpha / 2)

    return TreatmentEffectResult(
        estimand="ATE",
        estimate=ate,
        std_error=se,
        ci_lower=ate - z * se,
        ci_upper=ate + z * se,
        p_value=p_val,
        n_obs=len(df),
    )


def estimate_atc_matched(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    match_pairs: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> TreatmentEffectResult:
    """Estimate ATC from matched data (matching controls to treated).

    Parameters
    ----------
    data : pd.DataFrame
    outcome, treatment : str
    match_pairs : pd.DataFrame
        With columns ``control_idx`` and ``treated_idx`` (reverse
        matching direction).
    alpha : float

    Returns
    -------
    TreatmentEffectResult
    """
    diffs = []
    for _, row in match_pairs.iterrows():
        c_id = row.get("control_idx")
        t_id = row.get("treated_idx")
        if c_id in data.index and t_id in data.index:
            y_c = float(data.loc[c_id, outcome])
            y_t = float(data.loc[t_id, outcome])
            diffs.append(y_t - y_c)  # ATC = E[Y(1) - Y(0) | D=0]

    if len(diffs) == 0:
        return TreatmentEffectResult(
            estimand="ATC",
            estimate=np.nan,
            std_error=np.nan,
            ci_lower=np.nan,
            ci_upper=np.nan,
            p_value=np.nan,
            n_obs=0,
        )

    diffs = np.array(diffs)
    atc = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
    t_val = atc / se if se > 0 else 0.0
    p_val = float(2 * stats.norm.sf(abs(t_val)))
    z = stats.norm.ppf(1 - alpha / 2)

    return TreatmentEffectResult(
        estimand="ATC",
        estimate=atc,
        std_error=se,
        ci_lower=atc - z * se,
        ci_upper=atc + z * se,
        p_value=p_val,
        n_obs=len(diffs),
    )


# ---------------------------------------------------------------------------
# Abadie-Imbens standard errors
# ---------------------------------------------------------------------------


def abadie_imbens_se(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    match_pairs: pd.DataFrame,
    *,
    n_matches: int = 1,
) -> float:
    r"""Abadie-Imbens (2006) standard error for matching estimators.

    Accounts for the fact that matching introduces correlation across
    matched observations.  Uses the conditional variance estimator:

    .. math::

        \hat V_{AI} = \frac{1}{n^2}\sum_i \bigl[\hat\sigma^2(X_i)
        + (K_M(i))^2\,\hat\sigma^2(X_i)\bigr]

    where :math:`K_M(i)` is the number of times unit *i* is used as
    a match.

    Parameters
    ----------
    data : pd.DataFrame
    outcome, treatment : str
    match_pairs : pd.DataFrame
    n_matches : int

    Returns
    -------
    float
        Abadie-Imbens SE.

    References
    ----------
    Abadie, A., & Imbens, G. W. (2006). Large sample properties of
    matching estimators for average treatment effects. *Econometrica*,
    74(1), 235--267.
    """
    df = data.dropna(subset=[outcome, treatment]).copy()
    n = len(df)
    y = df[outcome].values.astype(float)
    d = df[treatment].values.astype(int)

    # Count how many times each unit is used as a match
    K = np.zeros(n)
    idx_to_pos = {idx: pos for pos, idx in enumerate(df.index)}

    for _, row in match_pairs.iterrows():
        c_idx = row["control_idx"]
        if c_idx in idx_to_pos:
            K[idx_to_pos[c_idx]] += 1

    # Estimate conditional variance using matched pairs
    sigma2 = np.zeros(n)
    for _, row in match_pairs.iterrows():
        t_id = row["treated_idx"]
        c_id = row["control_idx"]
        if t_id in idx_to_pos and c_id in idx_to_pos:
            t_pos = idx_to_pos[t_id]
            c_pos = idx_to_pos[c_id]
            # Simple variance estimate: half the squared difference
            diff2 = (y[t_pos] - y[c_pos]) ** 2 / 2
            sigma2[t_pos] = diff2
            sigma2[c_pos] = diff2

    # AI variance
    V = 0.0
    for i in range(n):
        V += (1 + K[i]) ** 2 * sigma2[i]
    V /= n**2

    return float(np.sqrt(max(V, 0.0)))


# ---------------------------------------------------------------------------
# Rosenbaum bounds (sensitivity analysis)
# ---------------------------------------------------------------------------


def rosenbaum_bounds(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    match_pairs: pd.DataFrame,
    *,
    gamma_range: Sequence[float] | None = None,
) -> pd.DataFrame:
    r"""Rosenbaum bounds for sensitivity to hidden bias.

    For each value of :math:`\Gamma` (the maximum odds ratio of
    differential treatment assignment due to an unobserved confounder),
    computes the upper and lower bounds on the p-value for the treatment
    effect.

    Parameters
    ----------
    data : pd.DataFrame
    outcome, treatment : str
    match_pairs : pd.DataFrame
    gamma_range : sequence of float, optional
        Values of :math:`\Gamma` to evaluate (default 1.0 to 3.0).

    Returns
    -------
    pd.DataFrame
        Columns: ``gamma``, ``p_lower``, ``p_upper``, ``significant_lower``,
        ``significant_upper``.

    References
    ----------
    Rosenbaum, P. R. (2002). *Observational Studies* (2nd ed.). Springer.
    """
    if gamma_range is None:
        gamma_range = [1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0]

    # Compute paired differences
    diffs = []
    for _, row in match_pairs.iterrows():
        t_id = row["treated_idx"]
        c_id = row["control_idx"]
        if t_id in data.index and c_id in data.index:
            diffs.append(float(data.loc[t_id, outcome]) - float(data.loc[c_id, outcome]))

    if len(diffs) == 0:
        return pd.DataFrame(columns=["gamma", "p_lower", "p_upper"])

    diffs = np.array(diffs)
    n_pairs = len(diffs)

    # Wilcoxon signed-rank approach
    abs_diffs = np.abs(diffs)
    ranks = stats.rankdata(abs_diffs)
    signs = np.sign(diffs)

    results = []
    for gamma in gamma_range:
        if gamma == 1.0:
            # Standard Wilcoxon test
            T_plus = float(np.sum(ranks[signs > 0]))
            E_T = n_pairs * (n_pairs + 1) / 4
            V_T = n_pairs * (n_pairs + 1) * (2 * n_pairs + 1) / 24
            z = (T_plus - E_T) / np.sqrt(max(V_T, 1e-10))
            p_val = float(2 * stats.norm.sf(abs(z)))
            results.append(
                {
                    "gamma": gamma,
                    "p_lower": p_val,
                    "p_upper": p_val,
                    "significant_lower": p_val < 0.05,
                    "significant_upper": p_val < 0.05,
                }
            )
        else:
            # Bounds on the probability of positive sign
            p_plus_upper = gamma / (1 + gamma)
            p_plus_lower = 1 / (1 + gamma)

            # Upper bound on p-value (worst case)
            E_upper = np.sum(ranks * p_plus_upper)
            V_upper = np.sum(ranks**2 * p_plus_upper * (1 - p_plus_upper))
            T_plus = float(np.sum(ranks[signs > 0]))
            z_upper = (T_plus - E_upper) / np.sqrt(max(V_upper, 1e-10))
            p_upper = float(stats.norm.sf(z_upper))

            # Lower bound
            E_lower = np.sum(ranks * p_plus_lower)
            V_lower = np.sum(ranks**2 * p_plus_lower * (1 - p_plus_lower))
            z_lower = (T_plus - E_lower) / np.sqrt(max(V_lower, 1e-10))
            p_lower = float(stats.norm.sf(z_lower))

            results.append(
                {
                    "gamma": gamma,
                    "p_lower": min(p_lower, p_upper),
                    "p_upper": max(p_lower, p_upper),
                    "significant_lower": min(p_lower, p_upper) < 0.05,
                    "significant_upper": max(p_lower, p_upper) < 0.05,
                }
            )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Doubly-robust estimation with matching
# ---------------------------------------------------------------------------


def doubly_robust_matching(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    covariates: list[str],
    *,
    ps: pd.Series | None = None,
    n_bootstrap: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
) -> TreatmentEffectResult:
    r"""Doubly-robust ATT estimation combining matching and regression.

    Matches on propensity score, then uses bias-corrected regression
    adjustment within matched pairs.  Consistent if either the
    propensity score model or the outcome model is correct.

    Parameters
    ----------
    data : pd.DataFrame
    outcome, treatment : str
    covariates : list of str
    ps : pd.Series, optional
    n_bootstrap : int
    seed : int
    alpha : float

    Returns
    -------
    TreatmentEffectResult
    """
    rng = np.random.default_rng(seed)
    df = data.dropna(subset=[outcome, treatment] + covariates).copy()

    # Match
    match_result = match_nearest_neighbor(df, treatment, covariates, n_neighbors=1, ps=ps)
    matched = match_result.matched_data

    # Outcome regression on controls
    c_mask = matched[treatment] == 0
    t_mask = matched[treatment] == 1
    X_c = matched.loc[c_mask, covariates].values.astype(float)
    y_c = matched.loc[c_mask, outcome].values.astype(float)

    from sklearn.linear_model import LinearRegression

    or_model = LinearRegression()
    or_model.fit(X_c, y_c)

    # Bias-corrected ATT
    X_t = matched.loc[t_mask, covariates].values.astype(float)
    y_t = matched.loc[t_mask, outcome].values.astype(float)
    y0_hat_t = or_model.predict(X_t)
    att_dr = float(np.mean(y_t - y0_hat_t))

    # Bootstrap SE
    n = len(df)
    boot_ests = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        df_b = df.iloc[idx].copy()
        df_b = df_b.reset_index(drop=True)
        try:
            mr = match_nearest_neighbor(df_b, treatment, covariates, n_neighbors=1)
            md = mr.matched_data
            cm = md[treatment] == 0
            tm = md[treatment] == 1
            if cm.sum() < 2 or tm.sum() < 2:
                continue
            lr = LinearRegression().fit(
                md.loc[cm, covariates].values.astype(float),
                md.loc[cm, outcome].values.astype(float),
            )
            y0h = lr.predict(md.loc[tm, covariates].values.astype(float))
            boot_ests.append(float(np.mean(md.loc[tm, outcome].values.astype(float) - y0h)))
        except Exception:
            continue

    se = float(np.std(boot_ests, ddof=1)) if len(boot_ests) > 1 else np.nan
    t_val = att_dr / se if se > 0 else 0.0
    p_val = float(2 * stats.norm.sf(abs(t_val)))
    z = stats.norm.ppf(1 - alpha / 2)

    return TreatmentEffectResult(
        estimand="ATT_DR",
        estimate=att_dr,
        std_error=se,
        ci_lower=att_dr - z * se,
        ci_upper=att_dr + z * se,
        p_value=p_val,
        n_obs=len(matched),
    )


# ---------------------------------------------------------------------------
# Matching with multiple treatments
# ---------------------------------------------------------------------------


def match_multi_treatment(
    data: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    *,
    reference_group: Any | None = None,
    method: str = "nearest_neighbor",
) -> dict[str, MatchResult]:
    """Matching with multiple (> 2) treatment groups.

    For each non-reference treatment level, matches treated units to
    the reference group using propensity scores from multinomial
    logistic regression.

    Parameters
    ----------
    data : pd.DataFrame
    treatment : str
        Column with multiple treatment levels.
    covariates : list of str
    reference_group : any, optional
        The reference (control) group level.  Defaults to the most
        frequent level.
    method : str
        ``"nearest_neighbor"`` or ``"mahalanobis"``.

    Returns
    -------
    dict
        Keys are treatment levels; values are :class:`MatchResult`.
    """
    df = data.dropna(subset=[treatment] + covariates).copy()
    levels = sorted(df[treatment].unique())

    if reference_group is None:
        reference_group = df[treatment].mode().iloc[0]

    results = {}
    for lvl in levels:
        if lvl == reference_group:
            continue

        # Binary comparison: lvl vs reference
        df_binary = df[df[treatment].isin([lvl, reference_group])].copy()
        df_binary["_treat_binary"] = (df_binary[treatment] == lvl).astype(int)

        if method == "mahalanobis":
            mr = match_mahalanobis(df_binary, "_treat_binary", covariates)
        else:
            mr = match_nearest_neighbor(df_binary, "_treat_binary", covariates)

        mr.details["treatment_level"] = lvl
        mr.details["reference_group"] = reference_group
        results[lvl] = mr

    return results


# ---------------------------------------------------------------------------
# Longitudinal / panel matching
# ---------------------------------------------------------------------------


def match_longitudinal(
    data: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    unit: str,
    time: str,
    treatment_time: str,
    *,
    n_pre_periods: int = 1,
    method: str = "nearest_neighbor",
) -> MatchResult:
    """Longitudinal matching for panel data.

    Matches treated and control units based on pre-treatment covariate
    values, allowing for time-varying covariates.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    treatment : str
    covariates : list of str
    unit : str
    time : str
    treatment_time : str
        Column indicating when treatment begins for each unit.
    n_pre_periods : int
        Number of pre-treatment periods to use for matching.
    method : str

    Returns
    -------
    MatchResult
    """
    df = data.copy()
    df["_treat_time"] = df[treatment_time].astype(float)

    # Determine pre-treatment covariates for each unit
    unit_features = []
    for u in df[unit].unique():
        u_data = df[df[unit] == u].sort_values(time)
        treat_t = u_data["_treat_time"].iloc[0]

        if np.isfinite(treat_t):
            pre_data = u_data[u_data[time] < treat_t].tail(n_pre_periods)
            is_treated = 1
        else:
            # For never-treated, use last n_pre_periods
            pre_data = u_data.tail(n_pre_periods)
            is_treated = 0

        if len(pre_data) == 0:
            continue

        features = {"_unit": u, "_treated": is_treated}
        for c in covariates:
            features[c] = float(pre_data[c].mean())
        unit_features.append(features)

    unit_df = pd.DataFrame(unit_features).set_index("_unit")

    if method == "mahalanobis":
        mr = match_mahalanobis(unit_df, "_treated", covariates)
    else:
        mr = match_nearest_neighbor(unit_df, "_treated", covariates)

    mr.method = f"longitudinal_{method}"
    return mr


# ---------------------------------------------------------------------------
# Matching quality assessment
# ---------------------------------------------------------------------------


def matching_quality(
    unmatched_data: pd.DataFrame,
    matched_data: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    *,
    weights: str | None = None,
) -> dict[str, Any]:
    """Comprehensive matching quality assessment.

    Compares balance before and after matching and computes summary
    metrics including percent bias reduction, number of balanced
    covariates, and overlap statistics.

    Parameters
    ----------
    unmatched_data, matched_data : pd.DataFrame
    treatment : str
    covariates : list of str
    weights : str, optional

    Returns
    -------
    dict
        Keys: ``balance_before``, ``balance_after``, ``bias_reduction``,
        ``pct_balanced_before``, ``pct_balanced_after``.
    """
    bal_before = balance_diagnostics(unmatched_data, treatment, covariates)
    bal_after = balance_diagnostics(matched_data, treatment, covariates, weights=weights)

    smd_before = bal_before.balance_table.set_index("covariate")["abs_smd"]
    smd_after = bal_after.balance_table.set_index("covariate")["abs_smd"]

    bias_reduction = {}
    for c in covariates:
        b = smd_before.get(c, np.nan)
        a = smd_after.get(c, np.nan)
        if b > 0 and not np.isnan(a):
            bias_reduction[c] = float(1 - a / b) * 100
        else:
            bias_reduction[c] = np.nan

    n_balanced_before = int((smd_before <= 0.1).sum())
    n_balanced_after = int((smd_after <= 0.1).sum())

    return {
        "balance_before": bal_before,
        "balance_after": bal_after,
        "bias_reduction": bias_reduction,
        "mean_bias_reduction": float(np.nanmean(list(bias_reduction.values()))),
        "pct_balanced_before": n_balanced_before / len(covariates) * 100 if covariates else 0,
        "pct_balanced_after": n_balanced_after / len(covariates) * 100 if covariates else 0,
        "n_obs_before": len(unmatched_data),
        "n_obs_after": len(matched_data),
    }


# ---------------------------------------------------------------------------
# Overlap diagnostics
# ---------------------------------------------------------------------------


def overlap_diagnostics(
    data: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    *,
    ps: pd.Series | None = None,
) -> dict[str, Any]:
    """Propensity score overlap diagnostics.

    Parameters
    ----------
    data : pd.DataFrame
    treatment : str
    covariates : list of str
    ps : pd.Series, optional

    Returns
    -------
    dict
        ``ps_summary`` (by group), ``overlap_region``, ``n_off_support``,
        ``effective_sample_size``.
    """
    df = data.dropna(subset=[treatment] + covariates).copy()
    if ps is None:
        ps = estimate_propensity_score(df, treatment, covariates)
    df["_ps"] = ps.reindex(df.index)

    t_ps = df.loc[df[treatment] == 1, "_ps"]
    c_ps = df.loc[df[treatment] == 0, "_ps"]

    overlap_lower = max(t_ps.min(), c_ps.min())
    overlap_upper = min(t_ps.max(), c_ps.max())

    on_support = (df["_ps"] >= overlap_lower) & (df["_ps"] <= overlap_upper)

    # IPW effective sample size
    ps_clip = np.clip(df["_ps"].values, 0.01, 0.99)
    d = df[treatment].values.astype(float)
    ipw_weights = d + (1 - d) * ps_clip / (1 - ps_clip)
    ess = float(ipw_weights.sum() ** 2 / (ipw_weights**2).sum())

    return {
        "ps_summary": df.groupby(treatment)["_ps"].describe(),
        "overlap_region": (float(overlap_lower), float(overlap_upper)),
        "n_off_support": int((~on_support).sum()),
        "pct_off_support": float((~on_support).mean() * 100),
        "effective_sample_size": ess,
    }
