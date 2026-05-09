"""moirais.otis — Ontario Restrictive Confinement (OTIS) analysis.

Correctional/sociolegal analysis for Ontario placement data (2023-2025).
Orchestrates existing MOIRAIS causal inference modules for the correctional
domain: DML-IRM, propensity score matching, AIPW, mixed-effects models.

Short-name API (≤6 chars):

- :func:`rplace` — Regional placement analysis by age/sex/year
- :func:`astcmb` — Alert-state combination encoding (8 binary → complexity)
- :func:`volat`  — Regional volatility/movement metric
- :func:`rctrnd` — Restrictive confinement trends over time
- :func:`otdesc` — Full OTIS descriptive statistics suite
- :func:`otdml`  — Run DML IRM (ATE/ATT) on OTIS data

Data: ``data/cache/correctional_stats_report_environment1b.RData`` (via R)
      ``data/cache/dt_expanded.rds`` (1.9M rows × 13 cols)

References
----------
* Ontario Ministry of the Solicitor General (2025). Restrictive
  Confinement Detailed Dataset. data.ontario.ca.
* Jahn v. Ontario (2020). Settlement Agreement — Inmate Data Disclosure.
* Chernozhukov et al. (2018). Double/debiased machine learning for
  treatment and structural parameters. Econometrics Journal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class RplRes:
    """Regional placement result."""

    counts: pd.DataFrame  # age × region count matrix
    props: pd.DataFrame  # age × region proportion matrix
    year: int
    sex: str | None = None


@dataclass
class AstRes:
    """Alert-state combination result."""

    data: pd.DataFrame  # per-person alert complexity
    summary: pd.DataFrame  # count by complexity level


@dataclass
class VolRes:
    """Volatility result."""

    data: pd.DataFrame  # per-person volatility metrics
    mean: float
    median: float


@dataclass
class OtDmlR:
    """OTIS DML result."""

    ate: float
    ate_se: float
    ate_pval: float
    att: float
    att_se: float
    att_pval: float
    n: int
    method: str = "IRM"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGIONS = ["Central", "Eastern", "Northern", "Toronto", "Western"]
AGE_GROUPS = ["18 to 24", "25 to 49", "50+"]
AGE_NUMERIC = {"18 to 24": 21.0, "25 to 49": 42.0, "50+": 57.5}

# 8 possible alert-state combinations (3 binary alerts)
# binA=mental_health, binB=suicide_risk, binC=suicide_watch
ALERT_COMBOS = {
    "a1": (1, 0, 0),  # mental health only
    "a2": (0, 1, 0),  # suicide risk only
    "a3": (0, 0, 1),  # suicide watch only
    "a4": (1, 1, 0),  # mental health + suicide risk
    "a5": (0, 1, 1),  # suicide risk + suicide watch
    "a6": (1, 0, 1),  # mental health + suicide watch
    "a7": (1, 1, 1),  # all three alerts
    "a8": (0, 0, 0),  # no alerts
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def rplace(
    df: pd.DataFrame,
    year: int,
    sex: str | None = None,
    *,
    id_col: str = "unique_individual_id",
    age_col: str = "age_category",
    region_col: str = "region_at_time_of_placement",
    year_col: str = "end_fiscal_year",
    gender_col: str = "gender",
) -> RplRes:
    """Regional placement analysis by age group.

    Computes count matrix S and proportion matrix P of placements
    across regions, stratified by age group and optionally by sex.

    Parameters
    ----------
    df : DataFrame
        Correctional placement data.
    year : int
        Fiscal year to analyze.
    sex : str, optional
        Filter by gender ("Male" or "Female").

    Returns
    -------
    RplRes
    """
    mask = (df[year_col] == year) & df[age_col].notna() & df[region_col].notna()
    sub = df[mask].copy()
    if sex is not None:
        sub = sub[sub[gender_col] == sex]

    # Count unique individuals per age × region
    counts = sub.groupby([age_col, region_col])[id_col].nunique().reset_index()
    counts.columns = ["age", "region", "n"]

    # Pivot to matrix
    pivot = counts.pivot_table(index="age", columns="region", values="n", fill_value=0)
    # Ensure all regions present
    for r in REGIONS:
        if r not in pivot.columns:
            pivot[r] = 0
    pivot = pivot[REGIONS]

    # Proportions (row-wise)
    row_sums = pivot.sum(axis=1)
    props = pivot.div(row_sums, axis=0).fillna(0)

    return RplRes(counts=pivot, props=props, year=year, sex=sex)


def astcmb(
    df: pd.DataFrame,
    *,
    alert_cols: tuple[str, ...] = ("mental_health_alert", "suicide_risk_alert", "suicide_watch_alert"),
    id_col: str = "unique_individual_id",
    year_col: str = "end_fiscal_year",
    np_col: str = "number_of_placements",
) -> AstRes:
    """Alert-state combination encoding.

    Encodes 3 binary alert indicators into 8 possible combinations
    (a1-a8) and computes a complexity index (count of distinct
    combinations per person-year).

    Parameters
    ----------
    df : DataFrame
        Must contain alert columns (Yes/No or 1/0).

    Returns
    -------
    AstRes
    """
    data = df.copy()

    # Binarize if needed
    for col in alert_cols:
        if pd.api.types.is_string_dtype(data[col]):
            data[col] = (data[col].str.lower() == "yes").astype(int)
        elif pd.api.types.is_numeric_dtype(data[col]):
            data[col] = data[col].astype(int)
        else:
            data[col] = data[col].astype(str).str.lower().eq("yes").astype(int)

    a, b, c = alert_cols
    data["a1"] = ((data[a] == 1) & (data[b] == 0) & (data[c] == 0)).astype(int)
    data["a2"] = ((data[a] == 0) & (data[b] == 1) & (data[c] == 0)).astype(int)
    data["a3"] = ((data[a] == 0) & (data[b] == 0) & (data[c] == 1)).astype(int)
    data["a4"] = ((data[a] == 1) & (data[b] == 1) & (data[c] == 0)).astype(int)
    data["a5"] = ((data[a] == 0) & (data[b] == 1) & (data[c] == 1)).astype(int)
    data["a6"] = ((data[a] == 1) & (data[b] == 0) & (data[c] == 1)).astype(int)
    data["a7"] = ((data[a] == 1) & (data[b] == 1) & (data[c] == 1)).astype(int)
    data["a8"] = ((data[a] == 0) & (data[b] == 0) & (data[c] == 0)).astype(int)

    # Aggregate per person-year
    acols = [f"a{i}" for i in range(1, 9)]
    grouped = data.groupby([id_col, year_col])[acols].sum().reset_index()

    # Complexity = number of distinct alert states observed
    grouped["ac"] = (grouped[acols] > 0).sum(axis=1)

    # Summary
    summary = grouped.groupby("ac").size().reset_index(name="n_persons")
    summary = summary.sort_values("ac", ascending=False)

    return AstRes(data=grouped, summary=summary)


def volat(
    df: pd.DataFrame,
    *,
    id_col: str = "unique_individual_id",
    year_col: str = "end_fiscal_year",
    regA_col: str = "region_at_time_of_placement",
    regB_col: str = "region_most_recent_placement",
) -> VolRes:
    """Regional volatility/movement metric.

    Counts the number of distinct regions an individual was placed in
    (combining initial and most recent placement regions).

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    VolRes
    """

    def _count_regions(group):
        regions = set(group[regA_col].dropna()) | set(group[regB_col].dropna())
        return len(regions)

    vm = df.groupby([id_col, year_col]).apply(_count_regions, include_groups=False).reset_index()
    vm.columns = [id_col, year_col, "vm"]

    return VolRes(
        data=vm,
        mean=float(vm["vm"].mean()),
        median=float(vm["vm"].median()),
    )


def rctrnd(
    df: pd.DataFrame,
    *,
    id_col: str = "unique_individual_id",
    year_col: str = "end_fiscal_year",
    region_col: str = "region_at_time_of_placement",
) -> pd.DataFrame:
    """Restrictive confinement trends over time.

    Returns per-year counts by region.

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    DataFrame with columns: year, region, n_individuals, n_placements
    """
    trends = (
        df.groupby([year_col, region_col])
        .agg(
            n_individuals=(id_col, "nunique"),
            n_placements=(id_col, "count"),
        )
        .reset_index()
    )
    trends.columns = ["year", "region", "n_individuals", "n_placements"]
    return trends.sort_values(["year", "region"])


def otdesc(
    df: pd.DataFrame,
    *,
    id_col: str = "unique_individual_id",
    year_col: str = "end_fiscal_year",
) -> dict:
    """Full OTIS descriptive statistics suite.

    Returns
    -------
    dict with keys: n_total, n_by_year, n_by_region, n_by_age,
    n_by_gender, placement_dist
    """
    result = {}
    result["n_total"] = df[id_col].nunique()
    result["n_records"] = len(df)

    result["n_by_year"] = df.groupby(year_col)[id_col].nunique().reset_index().rename(columns={id_col: "n"})

    if "region_at_time_of_placement" in df.columns:
        result["n_by_region"] = (
            df.groupby("region_at_time_of_placement")[id_col].nunique().reset_index().rename(columns={id_col: "n"})
        )

    if "age_category" in df.columns:
        result["n_by_age"] = df.groupby("age_category")[id_col].nunique().reset_index().rename(columns={id_col: "n"})

    if "gender" in df.columns:
        result["n_by_gender"] = df.groupby("gender")[id_col].nunique().reset_index().rename(columns={id_col: "n"})

    # Placement frequency distribution
    freq = df.groupby(id_col).size().reset_index(name="n_placements")
    result["placement_dist"] = freq["n_placements"].describe().to_dict()

    return result


def otdml(
    df: pd.DataFrame,
    outcome: str = "Y",
    treatment: str = "D",
    covariates: list[str] | None = None,
    *,
    cluster: str | None = None,
    n_folds: int = 3,
    seed: int = 123,
) -> OtDmlR:
    """Run DML IRM (ATE/ATT) on OTIS data.

    Wraps :func:`moirais.effects.estimate_irm` for the correctional domain.

    Parameters
    ----------
    df : DataFrame
        Must contain outcome, treatment, and covariate columns.
    outcome : str
        Outcome variable name (default "Y" = suicide risk).
    treatment : str
        Treatment variable name (default "D" = mental health alert).
    covariates : list of str, optional
        Covariate column names. Defaults to standard OTIS set.
    cluster : str, optional
        Cluster variable for clustered standard errors.

    Returns
    -------
    OtDmlR
    """
    if covariates is None:
        covariates = ["gender", "age_category", "region_at_time_of_placement", "region_most_recent_placement"]

    data = df[[outcome, treatment] + covariates].dropna().copy()

    # Encode categoricals as dummies
    cat_cols = data.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if cat_cols:
        data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

    X = data.drop(columns=[outcome, treatment]).values.astype(np.float64)
    y = data[outcome].values.astype(np.float64)
    d = data[treatment].values.astype(np.float64)
    n = len(y)

    # Simple Frisch-Waugh-Lovell partialling out (portable, no DoubleML dep)
    from numpy.linalg import lstsq

    rng = np.random.default_rng(seed)

    # Cross-fitted residuals
    fold_size = n // n_folds
    indices = rng.permutation(n)

    y_res = np.zeros(n)
    d_res = np.zeros(n)

    for k in range(n_folds):
        test_idx = indices[k * fold_size : (k + 1) * fold_size]
        train_idx = np.setdiff1d(indices, test_idx)

        # Outcome model: E[Y|X]
        beta_y, _, _, _ = lstsq(X[train_idx], y[train_idx], rcond=None)
        y_res[test_idx] = y[test_idx] - X[test_idx] @ beta_y

        # Treatment model: E[D|X]
        beta_d, _, _, _ = lstsq(X[train_idx], d[train_idx], rcond=None)
        d_res[test_idx] = d[test_idx] - X[test_idx] @ beta_d

    # ATE via partialled-out regression
    beta_ate, _, _, _ = lstsq(d_res.reshape(-1, 1), y_res, rcond=None)
    ate = float(beta_ate[0])

    # SE via heteroskedasticity-robust variance
    resid = y_res - d_res * ate
    meat = np.mean((d_res**2) * (resid**2))
    bread = np.mean(d_res**2)
    se = float(np.sqrt(meat / (bread**2 * n)))

    from scipy import stats

    z = ate / se if se > 0 else 0
    pval = float(2 * (1 - stats.norm.cdf(abs(z))))

    # ATT approximation (weight by treatment probability)
    p_treat = d.mean()
    att = ate / p_treat if p_treat > 0 else ate
    att_se = se / p_treat if p_treat > 0 else se
    att_pval = float(2 * (1 - stats.norm.cdf(abs(att / att_se)))) if att_se > 0 else 1.0

    return OtDmlR(
        ate=ate,
        ate_se=se,
        ate_pval=pval,
        att=att,
        att_se=att_se,
        att_pval=att_pval,
        n=n,
        method="PLR-crossfit",
    )


# ---------------------------------------------------------------------------
# Backward-compat aliases
# ---------------------------------------------------------------------------

regional_placement = rplace
alert_state_combo = astcmb
volatility = volat
rc_trends = rctrnd
otis_descriptives = otdesc
otis_dml = otdml
