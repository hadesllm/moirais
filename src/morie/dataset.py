"""
Dataset-agnostic analysis engine: ingest, profile, and plan analyses for any dataset.

This module provides tools to ingest datasets in multiple formats (CSV, TSV,
Excel, Parquet, JSON) without prior schema knowledge, infer levels of
measurement (NOIR: Nominal, Ordinal, Interval, Ratio) for each column,
auto-detect key variable roles (treatment, outcome, weight, stratum, cluster),
produce comprehensive dataset profiles, and suggest analysis plans.

The measurement-level inference follows Stevens' (1946) taxonomy as refined
for applied statistical practice:

- **Nominal** --- unordered categories (e.g., province, gender)
- **Ordinal** --- ordered categories (e.g., Likert scales, education level)
- **Interval** --- numeric with no true zero (e.g., temperature, year, index)
- **Ratio** --- numeric with true zero (e.g., count, weight, income, eBAC)

References
----------
Stevens, S. S. (1946). On the theory of scales of measurement.
*Science*, 103(2684), 677--680. https://doi.org/10.1126/science.103.2684.677

Velleman, P. F., & Wilkinson, L. (1993). Nominal, ordinal, interval, and
ratio typologies are misleading. *The American Statistician*, 47(1), 65--72.
https://doi.org/10.1080/00031305.1993.10475938
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Measurement level taxonomy
# ---------------------------------------------------------------------------


class MeasurementLevel(Enum):
    """Stevens' levels of measurement (NOIR taxonomy).

    .. list-table::
       :widths: 20 40 40
       :header-rows: 1

       * - Level
         - Properties
         - Example variables
       * - NOMINAL
         - Identity only (= / !=)
         - Province, gender, treatment indicator
       * - ORDINAL
         - Identity + order (< / >)
         - Likert scale, education level, income bracket
       * - INTERVAL
         - Identity + order + equal intervals, no true zero
         - Temperature (C), year, standardized score
       * - RATIO
         - Identity + order + equal intervals + true zero
         - Count, weight, income, eBAC

    References
    ----------
    Stevens, S. S. (1946). On the theory of scales of measurement.
    *Science*, 103(2684), 677--680.
    """

    NOMINAL = "nominal"
    ORDINAL = "ordinal"
    INTERVAL = "interval"
    RATIO = "ratio"


# ---------------------------------------------------------------------------
# Column-level profile
# ---------------------------------------------------------------------------


@dataclass
class ColumnProfile:
    """Profile of a single column in the dataset.

    Attributes
    ----------
    name : str
        Column name.
    dtype : str
        Pandas dtype as a string.
    level : MeasurementLevel
        Inferred Stevens measurement level.
    n_unique : int
        Number of distinct non-null values.
    missing_pct : float
        Percentage of missing values (0.0--100.0).
    is_binary : bool
        True if column has exactly two unique non-null values.
    is_constant : bool
        True if column has at most one unique non-null value.
    suggested_role : str
        One of: ``"treatment"``, ``"outcome"``, ``"covariate"``,
        ``"weight"``, ``"id"``, ``"stratum"``, ``"cluster"``, or
        ``"unknown"``.
    summary_stats : dict
        For numeric columns: ``{mean, std, min, q25, median, q75, max}``.
        For categorical columns: ``{top_counts}`` as a dict of value->count.
    """

    name: str
    dtype: str
    level: MeasurementLevel
    n_unique: int
    missing_pct: float
    is_binary: bool
    is_constant: bool
    suggested_role: str
    summary_stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dataset-level profile
# ---------------------------------------------------------------------------


@dataclass
class DatasetProfile:
    """Full profile of a dataset including all column profiles and summary diagnostics.

    Attributes
    ----------
    n_rows : int
        Number of rows.
    n_cols : int
        Number of columns.
    columns : dict[str, ColumnProfile]
        Per-column profiles keyed by column name.
    suggested_treatment : str | None
        Best-guess treatment column, or None if not detected.
    suggested_outcome : str | None
        Best-guess outcome column, or None if not detected.
    suggested_weights : str | None
        Best-guess survey-weight column, or None if not detected.
    """

    n_rows: int
    n_cols: int
    columns: dict[str, ColumnProfile]
    suggested_treatment: str | None = None
    suggested_outcome: str | None = None
    suggested_weights: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the profile to a plain dictionary.

        Returns
        -------
        dict
            Nested dictionary representation suitable for JSON serialization.
        """
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "suggested_treatment": self.suggested_treatment,
            "suggested_outcome": self.suggested_outcome,
            "suggested_weights": self.suggested_weights,
            "columns": {
                name: {
                    "name": cp.name,
                    "dtype": cp.dtype,
                    "level": cp.level.value,
                    "n_unique": cp.n_unique,
                    "missing_pct": cp.missing_pct,
                    "is_binary": cp.is_binary,
                    "is_constant": cp.is_constant,
                    "suggested_role": cp.suggested_role,
                    "summary_stats": cp.summary_stats,
                }
                for name, cp in self.columns.items()
            },
        }

    def summary_table(self) -> str:
        """Return a human-readable summary table.

        Uses :mod:`rich` ``Table`` if available; falls back to a plain-text
        tabular layout otherwise.

        Returns
        -------
        str
            Formatted summary string.
        """
        try:
            from io import StringIO

            from rich.console import Console
            from rich.table import Table

            table = Table(
                title=f"Dataset Profile  ({self.n_rows} rows x {self.n_cols} cols)",
                show_lines=True,
            )
            table.add_column("Column", style="bold cyan")
            table.add_column("dtype")
            table.add_column("Level")
            table.add_column("Unique")
            table.add_column("Missing%")
            table.add_column("Binary")
            table.add_column("Role")

            for cp in self.columns.values():
                table.add_row(
                    cp.name,
                    cp.dtype,
                    cp.level.value,
                    str(cp.n_unique),
                    f"{cp.missing_pct:.1f}",
                    str(cp.is_binary),
                    cp.suggested_role,
                )

            buf = StringIO()
            console = Console(file=buf, width=120)
            console.print(table)
            return buf.getvalue()
        except ImportError:
            lines = [
                f"Dataset Profile  ({self.n_rows} rows x {self.n_cols} cols)",
                "-" * 100,
                f"{'Column':<30} {'dtype':<12} {'Level':<10} {'Unique':>7} {'Miss%':>7} {'Binary':>7} {'Role':<15}",
                "-" * 100,
            ]
            for cp in self.columns.values():
                lines.append(
                    f"{cp.name:<30} {cp.dtype:<12} {cp.level.value:<10} {cp.n_unique:>7} "
                    f"{cp.missing_pct:>6.1f}% {str(cp.is_binary):>7} {cp.suggested_role:<15}"
                )
            return "\n".join(lines)


# ---------------------------------------------------------------------------
# Heuristic patterns for role detection
# ---------------------------------------------------------------------------

_TREATMENT_PATTERNS = re.compile(
    r"(treat|cannabis|drug|alcohol|interven|expos|assign|smok|vaccin|medic)",
    re.IGNORECASE,
)

_OUTCOME_PATTERNS = re.compile(
    r"(outcome|result|response|y_|_freq|_harm|_drink|disorder|death|mortal|"
    r"surviv|event|diagnos|prevalence|incidence|hospitali|readmit|relapse)",
    re.IGNORECASE,
)

_WEIGHT_PATTERNS = re.compile(
    r"(weight|^wt$|^wt_|_wt$|^pw$|^pw_|_pw$|survey_wt|wtpumf|samp.*wt|ipw|iptw)",
    re.IGNORECASE,
)

_STRATUM_PATTERNS = re.compile(
    r"(strat|stratum|_strata$)",
    re.IGNORECASE,
)

_CLUSTER_PATTERNS = re.compile(
    r"(cluster|psu|^clust_|_clust$)",
    re.IGNORECASE,
)

_ID_PATTERNS = re.compile(
    r"(^id$|_id$|^id_|^index$|^rowid|^record|^uid$|^caseid)",
    re.IGNORECASE,
)

_INTERVAL_NAME_PATTERNS = re.compile(
    r"(year|index|score|temperature|temp_|_temp$|date|time|latitude|longitude)",
    re.IGNORECASE,
)

_ORDINAL_NAME_PATTERNS = re.compile(
    r"(likert|rank|grade|level|stage|scale|class|quartile|quintile|decile|"
    r"tercile|rating|satisfaction|severity|frequency_cat|education|income_group|"
    r"age_group|health)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Core inference functions
# ---------------------------------------------------------------------------


def infer_measurement_level(
    series: pd.Series,
    *,
    ordinal_threshold: int = 10,
) -> MeasurementLevel:
    """Infer the Stevens measurement level (NOIR) for a single column.

    Detection rules (applied in order):

    1. **dtype object / category + ordinal name heuristic + n_unique <=
       ordinal_threshold**: ``ORDINAL``
    2. **dtype object / category** (default): ``NOMINAL``
    3. **dtype bool**: ``NOMINAL``
    4. **dtype int/float + is_binary (n_unique <= 2)**: ``NOMINAL`` (binary)
    5. **dtype int/float + n_unique <= 20 + ordinal name heuristic**: ``ORDINAL``
    6. **dtype float + interval name heuristic**: ``INTERVAL``
    7. **dtype float**: ``RATIO``
    8. **dtype int + min >= 0**: ``RATIO``
    9. **dtype int + min < 0**: ``INTERVAL``
    10. **datetime**: ``INTERVAL``

    Parameters
    ----------
    series : pandas.Series
        Column data to classify.
    ordinal_threshold : int, optional
        Maximum number of unique values for a categorical column to be
        considered ordinal (default 10).

    Returns
    -------
    MeasurementLevel
        Inferred measurement level.

    Examples
    --------
    >>> import pandas as pd
    >>> infer_measurement_level(pd.Series(["M", "F", "M", "F"]))
    <MeasurementLevel.NOMINAL: 'nominal'>

    >>> infer_measurement_level(pd.Series([1, 2, 3, 4, 5], name="likert_q1"))
    <MeasurementLevel.ORDINAL: 'ordinal'>
    """
    col_name = str(series.name or "")
    non_null = series.dropna()
    n_unique = int(non_null.nunique())

    # Categorical / object / string dtypes (includes pandas 2.x StringDtype)
    _is_text = (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )
    if _is_text:
        if n_unique <= ordinal_threshold and _ORDINAL_NAME_PATTERNS.search(col_name):
            return MeasurementLevel.ORDINAL
        return MeasurementLevel.NOMINAL

    # Boolean
    if pd.api.types.is_bool_dtype(series):
        return MeasurementLevel.NOMINAL

    # Numeric dtypes
    if pd.api.types.is_numeric_dtype(series):
        # Binary indicator
        if n_unique <= 2:
            return MeasurementLevel.NOMINAL

        # Ordinal heuristic for low-cardinality integer-like columns
        if n_unique <= 20 and _ORDINAL_NAME_PATTERNS.search(col_name):
            return MeasurementLevel.ORDINAL

        # Float columns
        if pd.api.types.is_float_dtype(series):
            if _INTERVAL_NAME_PATTERNS.search(col_name):
                return MeasurementLevel.INTERVAL
            return MeasurementLevel.RATIO

        # Integer columns
        if len(non_null) > 0:
            min_val = non_null.min()
            if min_val >= 0:
                return MeasurementLevel.RATIO
            return MeasurementLevel.INTERVAL
        return MeasurementLevel.RATIO

    # Datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return MeasurementLevel.INTERVAL

    return MeasurementLevel.NOMINAL


def _detect_role(series: pd.Series, *, level: MeasurementLevel) -> str:
    """Detect the suggested epidemiological role of a column.

    Parameters
    ----------
    series : pandas.Series
        Column data.
    level : MeasurementLevel
        Already-inferred measurement level.

    Returns
    -------
    str
        One of ``"treatment"``, ``"outcome"``, ``"weight"``, ``"id"``,
        ``"stratum"``, ``"cluster"``, or ``"covariate"``.
    """
    col_name = str(series.name or "")
    non_null = series.dropna()
    n_unique = int(non_null.nunique())
    is_binary = n_unique == 2

    # ID detection (prioritize to avoid misclassifying IDs as treatments)
    if _ID_PATTERNS.search(col_name):
        return "id"

    # Weight detection
    if _WEIGHT_PATTERNS.search(col_name) and pd.api.types.is_numeric_dtype(series):
        return "weight"

    # Stratum detection
    if _STRATUM_PATTERNS.search(col_name):
        return "stratum"

    # Cluster detection
    if _CLUSTER_PATTERNS.search(col_name):
        return "cluster"

    # Treatment detection: binary + name matches treatment patterns
    if is_binary and _TREATMENT_PATTERNS.search(col_name):
        return "treatment"

    # Outcome detection: name matches outcome patterns
    if _OUTCOME_PATTERNS.search(col_name):
        return "outcome"

    # Treatment fallback: any binary 0/1 column whose name suggests treatment
    if _TREATMENT_PATTERNS.search(col_name):
        return "treatment"

    return "covariate"


def _summarize_column(series: pd.Series, *, level: MeasurementLevel) -> dict:
    """Compute summary statistics appropriate for the column's measurement level.

    Parameters
    ----------
    series : pandas.Series
        Column data.
    level : MeasurementLevel
        Inferred measurement level.

    Returns
    -------
    dict
        Summary statistics.
    """
    non_null = series.dropna()

    if level in (MeasurementLevel.INTERVAL, MeasurementLevel.RATIO):
        if pd.api.types.is_numeric_dtype(non_null) and len(non_null) > 0:
            return {
                "mean": float(non_null.mean()),
                "std": float(non_null.std()),
                "min": float(non_null.min()),
                "q25": float(non_null.quantile(0.25)),
                "median": float(non_null.median()),
                "q75": float(non_null.quantile(0.75)),
                "max": float(non_null.max()),
            }
    # Categorical / ordinal / nominal / binary
    if len(non_null) > 0:
        top_counts = non_null.value_counts().head(10).to_dict()
        return {"top_counts": {str(k): int(v) for k, v in top_counts.items()}}
    return {"top_counts": {}}


# ---------------------------------------------------------------------------
# Dataset profiling
# ---------------------------------------------------------------------------


def profile_dataset(
    df: pd.DataFrame,
    *,
    hint_treatment: str | None = None,
    hint_outcome: str | None = None,
    hint_weights: str | None = None,
    ordinal_threshold: int = 10,
    binary_threshold: int = 2,
) -> DatasetProfile:
    """Fully profile a DataFrame without prior schema knowledge.

    Walks every column, infers its measurement level and epidemiological role,
    computes summary statistics, and auto-detects likely treatment, outcome,
    and survey-weight columns.  User-supplied hints override heuristic
    detection.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    hint_treatment : str, optional
        If provided, use this column as the treatment indicator.
    hint_outcome : str, optional
        If provided, use this column as the outcome.
    hint_weights : str, optional
        If provided, use this column as the survey weight.
    ordinal_threshold : int, optional
        Maximum unique values for categorical columns to be classified as
        ordinal (default 10).
    binary_threshold : int, optional
        Maximum unique values for a column to be considered binary
        (default 2).

    Returns
    -------
    DatasetProfile
        Complete dataset profile.

    Raises
    ------
    TypeError
        If *df* is not a pandas DataFrame.
    ValueError
        If *df* has zero rows or zero columns.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "treatment": [0, 1, 0, 1],
    ...     "outcome": [1.2, 3.4, 2.1, 4.5],
    ...     "age": [25, 30, 35, 40],
    ... })
    >>> profile = profile_dataset(df)
    >>> profile.n_rows
    4
    >>> profile.suggested_treatment
    'treatment'
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError(f"DataFrame must have at least one row and one column; got shape {df.shape}")

    columns: dict[str, ColumnProfile] = {}
    treatment_candidates: list[tuple[str, int]] = []  # (name, score)
    outcome_candidates: list[tuple[str, int]] = []
    weight_candidates: list[str] = []

    for col_name in df.columns:
        series = df[col_name]
        level = infer_measurement_level(series, ordinal_threshold=ordinal_threshold)
        role = _detect_role(series, level=level)

        non_null = series.dropna()
        n_unique = int(non_null.nunique())
        missing_pct = float(series.isna().mean() * 100.0)
        is_binary = n_unique == binary_threshold
        is_constant = n_unique <= 1
        summary = _summarize_column(series, level=level)

        cp = ColumnProfile(
            name=col_name,
            dtype=str(series.dtype),
            level=level,
            n_unique=n_unique,
            missing_pct=missing_pct,
            is_binary=is_binary,
            is_constant=is_constant,
            suggested_role=role,
            summary_stats=summary,
        )
        columns[col_name] = cp

        # Accumulate candidates for global role assignment
        if role == "treatment":
            score = 2 if is_binary else 1
            treatment_candidates.append((col_name, score))
        elif role == "outcome":
            score = 2 if pd.api.types.is_numeric_dtype(series) else 1
            outcome_candidates.append((col_name, score))
        elif role == "weight":
            weight_candidates.append(col_name)

    # Resolve best candidates (user hints override)
    suggested_treatment = hint_treatment
    if suggested_treatment is None and treatment_candidates:
        treatment_candidates.sort(key=lambda x: x[1], reverse=True)
        suggested_treatment = treatment_candidates[0][0]

    suggested_outcome = hint_outcome
    if suggested_outcome is None and outcome_candidates:
        outcome_candidates.sort(key=lambda x: x[1], reverse=True)
        suggested_outcome = outcome_candidates[0][0]

    suggested_weights = hint_weights
    if suggested_weights is None and weight_candidates:
        suggested_weights = weight_candidates[0]

    return DatasetProfile(
        n_rows=df.shape[0],
        n_cols=df.shape[1],
        columns=columns,
        suggested_treatment=suggested_treatment,
        suggested_outcome=suggested_outcome,
        suggested_weights=suggested_weights,
    )


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------


def load_dataset(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    **read_kwargs: Any,
) -> pd.DataFrame:
    """Load a dataset from CSV, TSV, Excel, Parquet, or JSON file.

    File format is detected from the extension.  Supported extensions:
    ``.csv``, ``.tsv``, ``.xlsx`` / ``.xls``, ``.parquet`` / ``.pq``,
    ``.json`` / ``.jsonl``.

    Parameters
    ----------
    path : str or Path
        File path to the dataset.
    encoding : str, optional
        Character encoding for text-based formats (default ``"utf-8"``).
    **read_kwargs
        Additional keyword arguments forwarded to the pandas reader.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file extension is not recognized.

    Examples
    --------
    >>> df = load_dataset("data.csv")
    >>> df = load_dataset("survey.xlsx", sheet_name="wave1")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    suffix = path.suffix.lower()
    logger.info("Loading dataset from %s (format: %s)", path, suffix)

    if suffix == ".csv":
        return pd.read_csv(path, encoding=encoding, **read_kwargs)
    elif suffix == ".tsv":
        return pd.read_csv(path, sep="\t", encoding=encoding, **read_kwargs)
    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, **read_kwargs)
    elif suffix in (".parquet", ".pq"):
        return pd.read_parquet(path, **read_kwargs)
    elif suffix == ".json":
        return pd.read_json(path, encoding=encoding, **read_kwargs)
    elif suffix == ".jsonl":
        return pd.read_json(path, lines=True, encoding=encoding, **read_kwargs)
    else:
        raise ValueError(
            f"Unsupported file extension: {suffix!r}. Supported: .csv, .tsv, .xlsx, .xls, .parquet, .pq, .json, .jsonl"
        )


# ---------------------------------------------------------------------------
# Analysis plan suggestion
# ---------------------------------------------------------------------------


def suggest_analysis_plan(profile: DatasetProfile) -> list[dict[str, Any]]:
    """Suggest an ordered analysis plan based on the detected variable types.

    Uses the inferred measurement levels, binary indicators, and detected
    treatment/outcome/weight columns to recommend epidemiological analyses.

    Parameters
    ----------
    profile : DatasetProfile
        Output from :func:`profile_dataset`.

    Returns
    -------
    list[dict[str, Any]]
        Ordered list of suggested analyses.  Each dict contains:

        - ``analysis`` (str): method identifier (e.g., ``"ipw_ate"``)
        - ``rationale`` (str): plain-English reason this analysis is suggested
        - ``required_vars`` (dict): mapping of role -> column name(s)

    Examples
    --------
    >>> plan = suggest_analysis_plan(profile)
    >>> plan[0]["analysis"]
    'descriptive_profile'
    """
    suggestions: list[dict[str, Any]] = []

    # Always suggest descriptive profiling
    suggestions.append(
        {
            "analysis": "descriptive_profile",
            "rationale": "Summarize variable distributions, missingness, and sample size before any modelling.",
            "required_vars": {"dataset": "all columns"},
        }
    )

    treatment = profile.suggested_treatment
    outcome = profile.suggested_outcome
    weights = profile.suggested_weights

    # Gather covariates
    covariates = [
        name for name, cp in profile.columns.items() if cp.suggested_role == "covariate" and not cp.is_constant
    ]

    if treatment and outcome:
        treatment_cp = profile.columns.get(treatment)
        outcome_cp = profile.columns.get(outcome)

        # Propensity score analysis
        if treatment_cp and treatment_cp.is_binary and covariates:
            suggestions.append(
                {
                    "analysis": "propensity_scores",
                    "rationale": (
                        f"Binary treatment '{treatment}' detected with covariates. "
                        "Estimate propensity scores for covariate balance."
                    ),
                    "required_vars": {
                        "treatment": treatment,
                        "covariates": covariates,
                    },
                }
            )

            # IPW ATE
            suggestions.append(
                {
                    "analysis": "ipw_ate",
                    "rationale": (f"Estimate ATE of '{treatment}' on '{outcome}' via inverse probability weighting."),
                    "required_vars": {
                        "treatment": treatment,
                        "outcome": outcome,
                        "covariates": covariates,
                    },
                }
            )

            # AIPW (doubly robust)
            suggestions.append(
                {
                    "analysis": "aipw",
                    "rationale": (
                        "Doubly-robust ATE estimation (AIPW) combining propensity "
                        "and outcome models for robustness to partial misspecification."
                    ),
                    "required_vars": {
                        "treatment": treatment,
                        "outcome": outcome,
                        "covariates": covariates,
                    },
                }
            )

            # ATT / ATC
            suggestions.append(
                {
                    "analysis": "att_atc",
                    "rationale": (
                        "Estimate ATT and ATC to understand treatment effects "
                        "on the treated and control populations separately."
                    ),
                    "required_vars": {
                        "treatment": treatment,
                        "outcome": outcome,
                        "covariates": covariates,
                    },
                }
            )

            # DML
            if outcome_cp and outcome_cp.level in (MeasurementLevel.INTERVAL, MeasurementLevel.RATIO):
                suggestions.append(
                    {
                        "analysis": "double_ml_plr",
                        "rationale": (
                            f"Continuous outcome '{outcome}' with binary treatment: "
                            "apply Double ML (PLR) with cross-fitting for Neyman-orthogonal ATE."
                        ),
                        "required_vars": {
                            "treatment": treatment,
                            "outcome": outcome,
                            "covariates": covariates,
                        },
                    }
                )

        # GATE if group variables exist
        group_candidates = [
            name
            for name, cp in profile.columns.items()
            if cp.suggested_role == "covariate"
            and cp.level in (MeasurementLevel.NOMINAL, MeasurementLevel.ORDINAL)
            and 2 < cp.n_unique <= 10
        ]
        if group_candidates and treatment_cp and treatment_cp.is_binary:
            suggestions.append(
                {
                    "analysis": "gate",
                    "rationale": (
                        f"Estimate Group Average Treatment Effects across "
                        f"{', '.join(group_candidates[:3])} to detect effect heterogeneity."
                    ),
                    "required_vars": {
                        "treatment": treatment,
                        "outcome": outcome,
                        "group_cols": group_candidates[:3],
                        "covariates": covariates,
                    },
                }
            )

    # Survey-weighted analysis
    if weights:
        suggestions.append(
            {
                "analysis": "survey_weighted_estimates",
                "rationale": (f"Survey weight column '{weights}' detected. Apply design-based weighted estimation."),
                "required_vars": {"weights": weights},
            }
        )

    return suggestions
