"""Output inspection and statistical verification for MORIE.

Provides two command families:

``morie inspect <path>``
    Browse output CSVs — show schema, row counts, summary statistics.

``morie verify <path>``
    Validate statistical outputs — check p-values, confidence intervals,
    standard errors, odds ratios, and other common statistical quantities
    for correctness.

Both commands work entirely offline (no LLM required) and render output
using ``rich`` when a TTY is detected.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .modules import MODULE_SPECS

# ---------------------------------------------------------------------------
# Column pattern matchers for statistical verification
# ---------------------------------------------------------------------------

_P_VALUE_PATTERNS = re.compile(r"(^p[_.\-]?val|^pvalue|^p$|_p$|_pval|significance)", re.IGNORECASE)
_CI_LOWER_PATTERNS = re.compile(r"(ci[_.\-]?lo|conf[_.\-]?lo|lower[_.\-]?ci|ci[_.\-]?2\.5|lower)", re.IGNORECASE)
_CI_UPPER_PATTERNS = re.compile(r"(ci[_.\-]?hi|conf[_.\-]?hi|upper[_.\-]?ci|ci[_.\-]?97\.5|upper)", re.IGNORECASE)
_SE_PATTERNS = re.compile(r"(^se$|std[_.\-]?err|standard[_.\-]?error|^se[_.\-])", re.IGNORECASE)
_OR_PATTERNS = re.compile(r"(odds[_.\-]?ratio|^or$|^or[_.\-]|[_.\-]or$)", re.IGNORECASE)
_ESTIMATE_PATTERNS = re.compile(
    r"(estimate|coef|coefficient|beta|effect|^ate$|^att$|^atc$|point[_.\-]?est)", re.IGNORECASE
)
_R_SQUARED_PATTERNS = re.compile(r"(r[_.\-]?squared|r2|rsq|pseudo[_.\-]?r2|adj[_.\-]?r2)", re.IGNORECASE)
_SAMPLE_SIZE_PATTERNS = re.compile(r"(^n$|^n[_.\-]obs|sample[_.\-]?size|^nobs$|n_total)", re.IGNORECASE)
_AIC_BIC_PATTERNS = re.compile(r"(^aic$|^bic$)", re.IGNORECASE)


def _match_columns(df: pd.DataFrame, pattern: re.Pattern) -> list[str]:
    """Return column names matching a regex pattern."""
    return [c for c in df.columns if pattern.search(c)]


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


@dataclass
class InspectionResult:
    """Summary of a single data file."""

    file_path: str
    rows: int
    columns: int
    column_names: list[str]
    dtypes: dict[str, str]
    missing_counts: dict[str, int]
    head: pd.DataFrame
    summary_stats: pd.DataFrame | None = None


def inspect_output(path: str | Path) -> InspectionResult:
    """Inspect a single output file.

    Parameters
    ----------
    path : str | Path
        Path to a CSV, TSV, Excel, or Parquet file.

    Returns
    -------
    InspectionResult
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    numeric_cols = df.select_dtypes(include=[np.number])
    summary = numeric_cols.describe().T if not numeric_cols.empty else None

    return InspectionResult(
        file_path=str(path),
        rows=len(df),
        columns=len(df.columns),
        column_names=list(df.columns),
        dtypes={c: str(df[c].dtype) for c in df.columns},
        missing_counts={c: int(df[c].isna().sum()) for c in df.columns},
        head=df.head(5),
        summary_stats=summary,
    )


def inspect_directory(
    directory: str | Path,
    *,
    module_name: str | None = None,
) -> list[InspectionResult]:
    """Inspect all CSV files in a directory.

    Parameters
    ----------
    directory : str | Path
        Directory to scan.
    module_name : str | None
        If provided, only inspect files expected by this module.

    Returns
    -------
    list[InspectionResult]
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    if module_name and module_name in MODULE_SPECS:
        expected = MODULE_SPECS[module_name].output_files
        files = [directory / f for f in expected if (directory / f).is_file()]
    else:
        files = sorted(directory.glob("*.csv"))

    return [inspect_output(f) for f in files]


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


@dataclass
class VerificationCheck:
    """A single verification check result."""

    name: str
    passed: bool
    message: str
    severity: str = "error"  # error | warning | info


@dataclass
class VerificationReport:
    """Verification report for a single file."""

    file_path: str
    checks: list[VerificationCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if no error-severity checks failed."""
        return all(c.passed for c in self.checks if c.severity == "error")

    @property
    def error_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "warning")


def verify_statistical_output(path: str | Path) -> VerificationReport:
    """Run statistical validation checks on an output file.

    Checks
    ------
    - p-values in [0, 1]
    - Standard errors >= 0
    - Odds ratios > 0
    - Confidence intervals: lower <= upper
    - Confidence intervals contain point estimate (when identifiable)
    - No NaN in critical columns (estimates, p-values)
    - Sample sizes > 0 and integer-valued
    - R-squared in [0, 1]
    - AIC/BIC are finite

    Parameters
    ----------
    path : str | Path
        Path to a CSV file with statistical output.

    Returns
    -------
    VerificationReport
    """
    path = Path(path)
    report = VerificationReport(file_path=str(path))

    if not path.is_file():
        report.checks.append(VerificationCheck("file_exists", False, f"File not found: {path}"))
        return report

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        report.checks.append(VerificationCheck("file_readable", False, f"Cannot read CSV: {exc}"))
        return report

    if df.empty:
        report.checks.append(VerificationCheck("not_empty", False, "File is empty", severity="warning"))
        return report

    report.checks.append(VerificationCheck("file_readable", True, f"{len(df)} rows, {len(df.columns)} columns"))

    # p-values in [0, 1]
    for col in _match_columns(df, _P_VALUE_PATTERNS):
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        bad = ((vals < 0) | (vals > 1)).sum()
        report.checks.append(
            VerificationCheck(
                f"p_value_range:{col}",
                bad == 0,
                f"{bad}/{len(vals)} values outside [0,1]" if bad else f"{len(vals)} values OK",
            )
        )

    # Standard errors >= 0
    for col in _match_columns(df, _SE_PATTERNS):
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        bad = (vals < 0).sum()
        report.checks.append(
            VerificationCheck(
                f"se_nonneg:{col}",
                bad == 0,
                f"{bad}/{len(vals)} negative SE values" if bad else f"{len(vals)} values OK",
            )
        )

    # Odds ratios > 0
    for col in _match_columns(df, _OR_PATTERNS):
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        bad = (vals <= 0).sum()
        report.checks.append(
            VerificationCheck(
                f"or_positive:{col}",
                bad == 0,
                f"{bad}/{len(vals)} non-positive OR values" if bad else f"{len(vals)} values OK",
            )
        )

    # CI: lower <= upper
    lower_cols = _match_columns(df, _CI_LOWER_PATTERNS)
    upper_cols = _match_columns(df, _CI_UPPER_PATTERNS)
    if lower_cols and upper_cols:
        lo = pd.to_numeric(df[lower_cols[0]], errors="coerce")
        hi = pd.to_numeric(df[upper_cols[0]], errors="coerce")
        valid = lo.notna() & hi.notna()
        if valid.sum() > 0:
            bad = (lo[valid] > hi[valid]).sum()
            report.checks.append(
                VerificationCheck(
                    "ci_order",
                    bad == 0,
                    f"{bad} rows with lower > upper CI" if bad else "CI bounds correctly ordered",
                )
            )

    # CI contains point estimate
    estimate_cols = _match_columns(df, _ESTIMATE_PATTERNS)
    if estimate_cols and lower_cols and upper_cols:
        est = pd.to_numeric(df[estimate_cols[0]], errors="coerce")
        lo = pd.to_numeric(df[lower_cols[0]], errors="coerce")
        hi = pd.to_numeric(df[upper_cols[0]], errors="coerce")
        valid = est.notna() & lo.notna() & hi.notna()
        if valid.sum() > 0:
            outside = ((est[valid] < lo[valid]) | (est[valid] > hi[valid])).sum()
            report.checks.append(
                VerificationCheck(
                    "ci_contains_estimate",
                    outside == 0,
                    f"{outside} estimates outside CI" if outside else "All estimates within CI",
                    severity="warning",
                )
            )

    # No NaN in critical columns
    for pattern, label in [
        (_ESTIMATE_PATTERNS, "estimate"),
        (_P_VALUE_PATTERNS, "p_value"),
    ]:
        for col in _match_columns(df, pattern):
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                report.checks.append(
                    VerificationCheck(
                        f"no_nan:{col}",
                        False,
                        f"{nan_count}/{len(df)} NaN values in {label} column",
                        severity="warning",
                    )
                )

    # Sample sizes > 0, integer
    for col in _match_columns(df, _SAMPLE_SIZE_PATTERNS):
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        non_positive = (vals <= 0).sum()
        non_integer = (vals != vals.astype(int)).sum()
        ok = non_positive == 0 and non_integer == 0
        msg_parts = []
        if non_positive:
            msg_parts.append(f"{non_positive} non-positive")
        if non_integer:
            msg_parts.append(f"{non_integer} non-integer")
        report.checks.append(
            VerificationCheck(
                f"sample_size:{col}",
                ok,
                ", ".join(msg_parts) if msg_parts else f"{len(vals)} values OK",
            )
        )

    # R-squared in [0, 1]
    for col in _match_columns(df, _R_SQUARED_PATTERNS):
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        bad = ((vals < 0) | (vals > 1)).sum()
        report.checks.append(
            VerificationCheck(
                f"r_squared_range:{col}",
                bad == 0,
                f"{bad}/{len(vals)} outside [0,1]" if bad else f"{len(vals)} values OK",
            )
        )

    # AIC/BIC finite
    for col in _match_columns(df, _AIC_BIC_PATTERNS):
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        non_finite = (~np.isfinite(vals)).sum()
        report.checks.append(
            VerificationCheck(
                f"ic_finite:{col}",
                non_finite == 0,
                f"{non_finite} non-finite values" if non_finite else f"{len(vals)} values OK",
                severity="warning",
            )
        )

    return report


def verify_directory(
    directory: str | Path,
    *,
    module_name: str | None = None,
) -> list[VerificationReport]:
    """Verify all CSV files in a directory.

    Parameters
    ----------
    directory : str | Path
        Directory to scan.
    module_name : str | None
        If provided, only verify files expected by this module.

    Returns
    -------
    list[VerificationReport]
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    if module_name and module_name in MODULE_SPECS:
        expected = MODULE_SPECS[module_name].output_files
        files = [directory / f for f in expected if (directory / f).is_file()]
    else:
        files = sorted(directory.glob("*.csv"))

    return [verify_statistical_output(f) for f in files]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_inspection(result: InspectionResult) -> None:
    """Print an inspection result to the terminal."""
    if sys.stdout.isatty():
        _render_inspection_rich(result)
    else:
        _render_inspection_plain(result)


def _render_inspection_rich(result: InspectionResult) -> None:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    # Header
    console.print(
        Panel(
            f"[bold]{result.file_path}[/bold]\n{result.rows} rows x {result.columns} columns",
            title="Inspection",
            border_style="cyan",
        )
    )

    # Column info
    col_table = Table(
        title="Columns",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
    )
    col_table.add_column("Name", style="bold")
    col_table.add_column("Type")
    col_table.add_column("Missing", justify="right")

    for col in result.column_names:
        missing = result.missing_counts.get(col, 0)
        style = "red" if missing > 0 else ""
        col_table.add_row(col, result.dtypes[col], str(missing), style=style)

    console.print(col_table)

    # Head
    if not result.head.empty:
        console.print("\n[bold]First 5 rows:[/bold]")
        head_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        for col in result.head.columns:
            head_table.add_column(str(col))
        for _, row in result.head.iterrows():
            head_table.add_row(*[str(v) for v in row])
        console.print(head_table)

    # Summary stats
    if result.summary_stats is not None:
        console.print("\n[bold]Summary statistics:[/bold]")
        stats_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        stats_table.add_column("Column", style="bold")
        for stat_col in result.summary_stats.columns:
            stats_table.add_column(str(stat_col), justify="right")
        for idx, row in result.summary_stats.iterrows():
            stats_table.add_row(str(idx), *[f"{v:.4g}" for v in row])
        console.print(stats_table)


def _render_inspection_plain(result: InspectionResult) -> None:
    print(f"File: {result.file_path}")
    print(f"Shape: {result.rows} rows x {result.columns} columns")
    print("Columns:")
    for col in result.column_names:
        missing = result.missing_counts.get(col, 0)
        miss_str = f" ({missing} missing)" if missing > 0 else ""
        print(f"  {col}: {result.dtypes[col]}{miss_str}")
    print()
    print("Head:")
    print(result.head.to_string(index=False))


def render_verification(report: VerificationReport) -> None:
    """Print a verification report to the terminal."""
    if sys.stdout.isatty():
        _render_verification_rich(report)
    else:
        _render_verification_plain(report)


def _render_verification_rich(report: VerificationReport) -> None:
    from rich import box
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(
        title=f"Verification: {report.file_path}",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Status", width=6, justify="center")
    table.add_column("Check", style="bold", min_width=28)
    table.add_column("Detail")

    for check in report.checks:
        if check.passed:
            status = "[green]  OK [/green]"
        elif check.severity == "error":
            status = "[red] FAIL[/red]"
        else:
            status = "[yellow] WARN[/yellow]"
        table.add_row(status, check.name, check.message)

    console.print(table)

    if report.passed:
        console.print("[green]All checks passed.[/green]")
    else:
        console.print(f"[red]{report.error_count} error(s), {report.warning_count} warning(s)[/red]")


def _render_verification_plain(report: VerificationReport) -> None:
    print(f"Verification: {report.file_path}")
    print("=" * 50)
    for check in report.checks:
        if check.passed:
            status = " OK "
        elif check.severity == "error":
            status = "FAIL"
        else:
            status = "WARN"
        print(f"  [{status}] {check.name:<30} {check.message}")
    print()
    if report.passed:
        print("All checks passed.")
    else:
        print(f"{report.error_count} error(s), {report.warning_count} warning(s)")
