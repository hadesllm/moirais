"""
Publication-ready table generation for epidemiological analysis.

Generates Table 1 (baseline characteristics), regression tables, odds ratio
tables, hazard ratio tables, correlation matrices, model comparison tables,
and ANOVA tables.  Supports output in multiple formats: pandas DataFrame,
LaTeX (booktabs), HTML, Markdown, plain text, and CSV.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

FormatTarget = Literal["dataframe", "latex", "html", "markdown", "text", "csv"]


def _fmt_num(x: float, digits: int = 2, apa: bool = False) -> str:
    """Format a number with fixed decimal places."""
    if not np.isfinite(x):
        return ""
    s = f"{x:.{digits}f}"
    if apa and abs(x) < 1 and s.startswith("0"):
        s = s.lstrip("0")
    elif apa and abs(x) < 1 and s.startswith("-0"):
        s = "-" + s[2:]
    return s


def _fmt_pval(p: float, digits: int = 3, apa: bool = False) -> str:
    """Format a p-value, with <0.001 convention."""
    if not np.isfinite(p):
        return ""
    if p < 10 ** (-digits):
        return f"<{'.' if apa else '0.'}{'0' * (digits - 1)}1"
    s = f"{p:.{digits}f}"
    if apa and s.startswith("0"):
        s = s.lstrip("0")
    return s


def _stars(p: float) -> str:
    """Return significance stars."""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _smd(mean1: float, mean2: float, sd1: float, sd2: float) -> float:
    """Compute standardised mean difference (Cohen's d variant)."""
    pooled_sd = np.sqrt((sd1**2 + sd2**2) / 2)
    if pooled_sd < 1e-12:
        return 0.0
    return (mean1 - mean2) / pooled_sd


# ---------------------------------------------------------------------------
# Footnote system
# ---------------------------------------------------------------------------


@dataclass
class FootnoteRegistry:
    """Accumulates footnotes and renders them in order."""

    _notes: list[str] = field(default_factory=list)
    _symbols: str = field(default="abcdefghijklmnopqrstuvwxyz")

    def add(self, text: str) -> str:
        """Add a footnote, return the symbol."""
        if text not in self._notes:
            self._notes.append(text)
        idx = self._notes.index(text)
        return self._symbols[idx % len(self._symbols)]

    def render(self, fmt: FormatTarget = "text") -> str:
        """Render all footnotes."""
        lines = []
        for i, note in enumerate(self._notes):
            sym = self._symbols[i % len(self._symbols)]
            if fmt == "latex":
                lines.append(f"\\textsuperscript{{{sym}}} {note}")
            elif fmt == "html":
                lines.append(f"<sup>{sym}</sup> {note}")
            else:
                lines.append(f"  {sym} {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------


def _to_format(
    df: pd.DataFrame,
    fmt: FormatTarget,
    *,
    title: str = "",
    footnotes: str = "",
    bold_significant: bool = False,
    spanning_headers: dict[str, list[str]] | None = None,
) -> str | pd.DataFrame:
    """Convert a DataFrame to the requested output format."""
    if fmt == "dataframe":
        return df

    if fmt == "csv":
        buf = StringIO()
        if title:
            buf.write(f"# {title}\n")
        df.to_csv(buf)
        if footnotes:
            buf.write(f"\n{footnotes}\n")
        return buf.getvalue()

    if fmt == "markdown":
        md = df.to_markdown(index=True)
        parts = []
        if title:
            parts.append(f"**{title}**\n")
        parts.append(md)
        if footnotes:
            parts.append(f"\n{footnotes}")
        return "\n".join(parts)

    if fmt == "text":
        parts = []
        if title:
            parts.append(title)
            parts.append("=" * len(title))
        parts.append(df.to_string())
        if footnotes:
            parts.append(f"\n{footnotes}")
        return "\n".join(parts)

    if fmt == "html":
        html = df.to_html(classes=["moirais-table", "table", "table-striped"], border=0, escape=False)
        parts = []
        if title:
            parts.append(f'<caption class="moirais-caption">{title}</caption>')
        parts.append(html)
        if footnotes:
            fn_html = footnotes.replace("\n", "<br>")
            parts.append(f'<div class="moirais-footnotes">{fn_html}</div>')

        # Spanning headers
        if spanning_headers:
            header_row = "<tr>"
            for span_label, cols in spanning_headers.items():
                header_row += (
                    f'<th colspan="{len(cols)}" style="text-align:center; border-bottom: 2px solid #333;">'
                    f"{span_label}</th>"
                )
            header_row += "</tr>"
            html = parts[0] if title else ""
            html += parts[-2] if len(parts) > 1 else parts[0]
            # Insert spanning header after <thead><tr>
            idx = html.find("<thead>")
            if idx >= 0:
                html = html[: idx + 7] + header_row + html[idx + 7 :]
            parts = [html]
            if footnotes:
                parts.append(f'<div class="moirais-footnotes">{footnotes.replace(chr(10), "<br>")}</div>')

        return "\n".join(parts)

    if fmt == "latex":
        n_cols = len(df.columns) + 1  # +1 for index
        col_spec = "l" + "c" * len(df.columns)
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
        ]
        if title:
            lines.append(f"\\caption{{{title}}}")
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")

        # Spanning headers
        if spanning_headers:
            parts_sh = []
            for span_label, cols in spanning_headers.items():
                parts_sh.append(f"\\multicolumn{{{len(cols)}}}{{c}}{{{span_label}}}")
            lines.append(" & ".join([""] + parts_sh) + " \\\\")
            # cmidrule
            col_idx = 2
            for span_label, cols in spanning_headers.items():
                lines.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + len(cols) - 1}}}")
                col_idx += len(cols)

        # Column headers
        header = " & ".join([str(df.index.name or "")] + [str(c) for c in df.columns])
        lines.append(header + " \\\\")
        lines.append("\\midrule")

        # Data rows
        for idx_val, row in df.iterrows():
            cells = [str(idx_val)]
            for v in row:
                cells.append(str(v))
            lines.append(" & ".join(cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        if footnotes:
            fn_tex = footnotes.replace("\n", " \\\\\n")
            lines.append(f"\\begin{{tablenotes}}\\small\n{fn_tex}\n\\end{{tablenotes}}")
        lines.append("\\end{table}")
        return "\n".join(lines)

    raise ValueError(f"Unsupported format: {fmt}")


# ---------------------------------------------------------------------------
# Table 1: baseline characteristics
# ---------------------------------------------------------------------------


def table1(
    data: pd.DataFrame,
    group_col: str | None = None,
    continuous_vars: list[str] | None = None,
    categorical_vars: list[str] | None = None,
    *,
    continuous_summary: str = "mean_sd",
    show_p: bool = True,
    show_smd: bool = True,
    show_missing: bool = True,
    weights: str | None = None,
    digits: int = 2,
    apa: bool = False,
    output_format: FormatTarget = "dataframe",
    title: str = "Table 1. Baseline Characteristics",
) -> pd.DataFrame | str:
    """
    Generate a Table 1 of baseline characteristics stratified by group.

    Parameters
    ----------
    data : DataFrame
        Input data with all variables.
    group_col : str or None
        Column name defining groups.  If ``None``, a single-column summary
        is produced.
    continuous_vars : list of str or None
        Continuous variables.  If ``None``, auto-detected as numeric
        non-group columns.
    categorical_vars : list of str or None
        Categorical variables.  If ``None``, auto-detected as object/
        category columns.
    continuous_summary : str
        Summary format for continuous variables:
        ``"mean_sd"``, ``"median_iqr"``, or ``"mean_ci"``.
    show_p : bool
        Include p-value column.
    show_smd : bool
        Include standardised mean difference column (requires exactly
        two groups).
    show_missing : bool
        Include a missing count column.
    weights : str or None
        Column name for survey weights.
    digits : int
        Number of decimal places.
    apa : bool
        Use APA-style formatting (leading-zero suppression for p-values).
    output_format : FormatTarget
        Output format.
    title : str
        Table title.

    Returns
    -------
    DataFrame or str
        Table in the requested format.

    References
    ----------
    Austin, P. C. (2011). An introduction to propensity score methods for
    reducing the effects of confounding in observational studies.
    *Multivariate Behavioral Research*, 46(3), 399-424.
    https://doi.org/10.1080/00273171.2011.568786
    """
    df = data.copy()
    footnotes = FootnoteRegistry()

    # Auto-detect variable types
    if continuous_vars is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        continuous_vars = [c for c in numeric_cols if c != group_col and c != weights]
    if categorical_vars is None:
        categorical_vars = [
            c for c in df.select_dtypes(include=["object", "category", "bool"]).columns if c != group_col
        ]

    if group_col is not None:
        groups = sorted(df[group_col].dropna().unique())
    else:
        groups = [None]

    # Build result rows
    rows: list[dict[str, str]] = []
    row_labels: list[str] = []

    # N row
    n_row: dict[str, str] = {}
    for g in groups:
        if g is not None:
            sub = df[df[group_col] == g]
        else:
            sub = df
        n_row[str(g) if g is not None else "Overall"] = str(len(sub))
    rows.append(n_row)
    row_labels.append("N")

    def _weighted_mean_sd(values: pd.Series, w: pd.Series | None):
        v = values.dropna()
        if w is not None:
            w_clean = w.loc[v.index]
            wm = np.average(v, weights=w_clean)
            variance = np.average((v - wm) ** 2, weights=w_clean)
            return wm, np.sqrt(variance)
        return v.mean(), v.std()

    # Continuous variables
    for var in continuous_vars:
        row: dict[str, str] = {}
        group_stats: list[tuple[float, float]] = []

        for g in groups:
            sub = df[df[group_col] == g] if g is not None else df
            vals = sub[var].dropna()
            w_vals = sub[weights] if weights else None

            if continuous_summary == "mean_sd":
                m, s = _weighted_mean_sd(vals, w_vals.loc[vals.index] if w_vals is not None else None)
                cell = f"{_fmt_num(m, digits)} ({_fmt_num(s, digits)})"
            elif continuous_summary == "median_iqr":
                med = vals.median()
                q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                cell = f"{_fmt_num(med, digits)} [{_fmt_num(q1, digits)}, {_fmt_num(q3, digits)}]"
                m, s = vals.mean(), vals.std()
            elif continuous_summary == "mean_ci":
                m, s = _weighted_mean_sd(vals, w_vals.loc[vals.index] if w_vals is not None else None)
                n_val = len(vals)
                se = s / np.sqrt(n_val) if n_val > 0 else 0
                lo, hi = m - 1.96 * se, m + 1.96 * se
                cell = f"{_fmt_num(m, digits)} ({_fmt_num(lo, digits)}, {_fmt_num(hi, digits)})"
            else:
                raise ValueError(f"Unknown continuous_summary: {continuous_summary}")

            col_name = str(g) if g is not None else "Overall"
            row[col_name] = cell
            group_stats.append((vals.mean(), vals.std()))

        # p-value
        if show_p and len(groups) >= 2:
            group_vals = [df[df[group_col] == g][var].dropna() for g in groups]
            if len(groups) == 2:
                stat, p = sp_stats.mannwhitneyu(group_vals[0], group_vals[1], alternative="two-sided")
            else:
                stat, p = sp_stats.kruskal(*group_vals)
            row["p-value"] = _fmt_pval(p, 3, apa)
            row[""] = _stars(p)  # stars column

        # SMD (only for 2 groups)
        if show_smd and len(groups) == 2:
            smd_val = _smd(group_stats[0][0], group_stats[1][0], group_stats[0][1], group_stats[1][1])
            row["SMD"] = _fmt_num(abs(smd_val), 3)

        # Missing
        if show_missing:
            n_miss = df[var].isna().sum()
            pct_miss = 100 * n_miss / len(df)
            row["Missing"] = f"{n_miss} ({_fmt_num(pct_miss, 1)}%)"

        rows.append(row)
        summary_label = {"mean_sd": "mean (SD)", "median_iqr": "median [IQR]", "mean_ci": "mean (95% CI)"}
        row_labels.append(f"{var}, {summary_label.get(continuous_summary, '')}")

    # Categorical variables
    for var in categorical_vars:
        categories = df[var].dropna().unique()
        # Header row
        rows.append({str(g) if g is not None else "Overall": "" for g in groups})
        row_labels.append(f"{var}, n (%)")

        for cat in sorted(categories, key=str):
            row_cat: dict[str, str] = {}
            group_counts: list[tuple[int, int]] = []

            for g in groups:
                sub = df[df[group_col] == g] if g is not None else df
                n_cat = (sub[var] == cat).sum()
                n_total = sub[var].notna().sum()
                pct = 100 * n_cat / n_total if n_total > 0 else 0
                col_name = str(g) if g is not None else "Overall"
                row_cat[col_name] = f"{n_cat} ({_fmt_num(pct, 1)}%)"
                group_counts.append((n_cat, n_total))

            rows.append(row_cat)
            row_labels.append(f"  {cat}")

        # p-value for categorical (chi-square)
        if show_p and len(groups) >= 2:
            contingency = pd.crosstab(df[var], df[group_col])
            try:
                chi2, p, _, _ = sp_stats.chi2_contingency(contingency)
            except ValueError:
                p = np.nan
            # Attach to the header row
            header_idx = len(rows) - len(categories) - 1
            rows[header_idx]["p-value"] = _fmt_pval(p, 3, apa)
            rows[header_idx][""] = _stars(p)

        # Missing
        if show_missing:
            n_miss = df[var].isna().sum()
            pct_miss = 100 * n_miss / len(df)
            header_idx = len(rows) - len(categories) - 1
            rows[header_idx]["Missing"] = f"{n_miss} ({_fmt_num(pct_miss, 1)}%)"

    # Build footnotes
    if show_p:
        footnotes.add("Continuous: Mann-Whitney U (2 groups) or Kruskal-Wallis; Categorical: Chi-square test")
    if show_smd:
        footnotes.add("SMD = Standardised Mean Difference (absolute value)")
    if weights:
        footnotes.add(f"Weighted by '{weights}'")

    result_df = pd.DataFrame(rows, index=row_labels)
    result_df.index.name = "Variable"

    return _to_format(result_df, output_format, title=title, footnotes=footnotes.render(output_format))


# ---------------------------------------------------------------------------
# Regression table
# ---------------------------------------------------------------------------


def regression_table(
    models: dict[str, Any],
    *,
    exponentiate: bool = False,
    show_ci: bool = True,
    show_stars: bool = True,
    confidence: float = 0.95,
    digits: int = 3,
    model_stats: list[str] | None = None,
    apa: bool = False,
    output_format: FormatTarget = "dataframe",
    title: str = "Regression Results",
    spanning_headers: dict[str, list[str]] | None = None,
) -> pd.DataFrame | str:
    """
    Multi-model regression table (side-by-side).

    Accepts fitted statsmodels results objects.  Supports linear, logistic,
    Poisson, and other GLM models.

    Parameters
    ----------
    models : dict
        ``{model_name: fitted_model}`` where *fitted_model* is a
        statsmodels results object with ``params``, ``bse``, ``pvalues``,
        ``conf_int()``, and model summary attributes.
    exponentiate : bool
        Exponentiate coefficients (e.g. for odds ratios or hazard ratios).
    show_ci : bool
        Include confidence intervals.
    show_stars : bool
        Include significance stars next to coefficients.
    confidence : float
        Confidence level for intervals.
    digits : int
        Decimal places.
    model_stats : list of str or None
        Model-level statistics to include.  Defaults to
        ``["nobs", "rsquared", "aic", "bic", "llf"]``.
    apa : bool
        APA formatting.
    output_format : FormatTarget
    title : str
    spanning_headers : dict or None
        Grouped column headers ``{span_label: [model_names]}``.

    Returns
    -------
    DataFrame or str

    Notes
    -----
    Each model column shows: coefficient (with stars), SE in parentheses,
    and optionally the CI on a separate line.
    """
    if model_stats is None:
        model_stats = ["nobs", "rsquared", "aic", "bic", "llf"]

    footnotes = FootnoteRegistry()
    model_names = list(models.keys())

    # Collect all parameter names
    all_params: list[str] = []
    for m in models.values():
        for p in m.params.index:
            if p not in all_params:
                all_params.append(p)

    alpha = 1 - confidence
    rows: list[dict[str, str]] = []
    row_labels: list[str] = []

    for param in all_params:
        coef_row: dict[str, str] = {}
        se_row: dict[str, str] = {}
        ci_row: dict[str, str] = {}

        for mname, model_obj in models.items():
            if param in model_obj.params.index:
                b = model_obj.params[param]
                se = model_obj.bse[param]
                p = model_obj.pvalues[param]
                ci = model_obj.conf_int(alpha=alpha).loc[param]

                if exponentiate:
                    b_display = np.exp(b)
                    ci_lo = np.exp(ci.iloc[0])
                    ci_hi = np.exp(ci.iloc[1])
                else:
                    b_display = b
                    ci_lo = ci.iloc[0]
                    ci_hi = ci.iloc[1]

                star_str = _stars(p) if show_stars else ""
                coef_row[mname] = f"{_fmt_num(b_display, digits, apa)}{star_str}"
                se_row[mname] = f"({_fmt_num(se, digits, apa)})"
                if show_ci:
                    ci_row[mname] = f"[{_fmt_num(ci_lo, digits)}, {_fmt_num(ci_hi, digits)}]"
            else:
                coef_row[mname] = ""
                se_row[mname] = ""
                if show_ci:
                    ci_row[mname] = ""

        rows.append(coef_row)
        row_labels.append(param)
        rows.append(se_row)
        row_labels.append("")
        if show_ci:
            rows.append(ci_row)
            row_labels.append("")

    # Model statistics
    stat_labels = {
        "nobs": "N",
        "rsquared": "R-squared",
        "rsquared_adj": "Adj. R-squared",
        "aic": "AIC",
        "bic": "BIC",
        "llf": "Log-Likelihood",
    }
    for stat in model_stats:
        stat_row: dict[str, str] = {}
        for mname, model_obj in models.items():
            val = getattr(model_obj, stat, None)
            if val is not None:
                if stat == "nobs":
                    stat_row[mname] = str(int(val))
                else:
                    stat_row[mname] = _fmt_num(float(val), digits)
            else:
                stat_row[mname] = ""
        rows.append(stat_row)
        row_labels.append(stat_labels.get(stat, stat))

    if show_stars:
        footnotes.add("* p < 0.05, ** p < 0.01, *** p < 0.001")
    footnotes.add(f"Standard errors in parentheses. CI level: {confidence * 100:.0f}%.")

    result_df = pd.DataFrame(rows, index=row_labels)
    result_df.index.name = ""
    return _to_format(
        result_df,
        output_format,
        title=title,
        footnotes=footnotes.render(output_format),
        spanning_headers=spanning_headers,
    )


# ---------------------------------------------------------------------------
# Odds ratio table
# ---------------------------------------------------------------------------


def odds_ratio_table(
    model: Any,
    *,
    confidence: float = 0.95,
    digits: int = 3,
    apa: bool = False,
    output_format: FormatTarget = "dataframe",
    title: str = "Odds Ratios",
) -> pd.DataFrame | str:
    """
    Odds ratio table from a fitted logistic regression model.

    Parameters
    ----------
    model : statsmodels results
        A fitted logistic regression (``Logit`` or ``GLM`` with binomial
        family).
    confidence : float
    digits : int
    apa : bool
    output_format : FormatTarget
    title : str

    Returns
    -------
    DataFrame or str
    """
    alpha = 1 - confidence
    ci = model.conf_int(alpha=alpha)
    params = model.params
    pvals = model.pvalues

    records = []
    for p in params.index:
        b = params[p]
        or_val = np.exp(b)
        or_lo = np.exp(ci.loc[p].iloc[0])
        or_hi = np.exp(ci.loc[p].iloc[1])
        pv = pvals[p]
        records.append(
            {
                "Variable": p,
                "OR": _fmt_num(or_val, digits, apa),
                f"{confidence * 100:.0f}% CI": f"({_fmt_num(or_lo, digits)}, {_fmt_num(or_hi, digits)})",
                "p-value": _fmt_pval(pv, digits, apa),
                "": _stars(pv),
            }
        )

    result_df = pd.DataFrame(records).set_index("Variable")
    fn = FootnoteRegistry()
    fn.add("OR = Odds Ratio. * p<0.05, ** p<0.01, *** p<0.001")
    return _to_format(result_df, output_format, title=title, footnotes=fn.render(output_format))


# ---------------------------------------------------------------------------
# Hazard ratio table
# ---------------------------------------------------------------------------


def hazard_ratio_table(
    params: pd.Series,
    se: pd.Series,
    pvalues: pd.Series,
    *,
    confidence: float = 0.95,
    digits: int = 3,
    apa: bool = False,
    output_format: FormatTarget = "dataframe",
    title: str = "Hazard Ratios",
) -> pd.DataFrame | str:
    """
    Hazard ratio table from Cox model parameters.

    Parameters
    ----------
    params : Series
        Log-hazard ratio coefficients (beta).
    se : Series
        Standard errors of coefficients.
    pvalues : Series
        P-values.
    confidence : float
    digits : int
    apa : bool
    output_format : FormatTarget
    title : str

    Returns
    -------
    DataFrame or str
    """
    z = sp_stats.norm.ppf(1 - (1 - confidence) / 2)
    records = []
    for var in params.index:
        b = params[var]
        s = se[var]
        hr = np.exp(b)
        hr_lo = np.exp(b - z * s)
        hr_hi = np.exp(b + z * s)
        pv = pvalues[var]
        records.append(
            {
                "Variable": var,
                "HR": _fmt_num(hr, digits, apa),
                f"{confidence * 100:.0f}% CI": f"({_fmt_num(hr_lo, digits)}, {_fmt_num(hr_hi, digits)})",
                "p-value": _fmt_pval(pv, digits, apa),
                "": _stars(pv),
            }
        )

    result_df = pd.DataFrame(records).set_index("Variable")
    fn = FootnoteRegistry()
    fn.add("HR = Hazard Ratio. * p<0.05, ** p<0.01, *** p<0.001")
    return _to_format(result_df, output_format, title=title, footnotes=fn.render(output_format))


# ---------------------------------------------------------------------------
# Correlation matrix table
# ---------------------------------------------------------------------------


def correlation_table(
    data: pd.DataFrame,
    *,
    method: str = "pearson",
    show_stars: bool = True,
    mask_diagonal: bool = True,
    digits: int = 3,
    output_format: FormatTarget = "dataframe",
    title: str = "Correlation Matrix",
) -> pd.DataFrame | str:
    """
    Correlation matrix with significance stars.

    Parameters
    ----------
    data : DataFrame
        Numeric columns to correlate.
    method : str
        ``"pearson"``, ``"spearman"``, or ``"kendall"``.
    show_stars : bool
        Annotate with significance stars.
    mask_diagonal : bool
        Replace diagonal with ``"-"``.
    digits : int
    output_format : FormatTarget
    title : str

    Returns
    -------
    DataFrame or str
    """
    numeric = data.select_dtypes(include="number")
    corr = numeric.corr(method=method)
    cols = corr.columns
    n_vars = len(cols)

    # p-values
    result = corr.copy().astype(str)
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                result.iloc[i, j] = "-" if mask_diagonal else _fmt_num(1.0, digits)
                continue
            valid = numeric[[cols[i], cols[j]]].dropna()
            r_val = corr.iloc[i, j]
            if len(valid) > 2:
                if method == "pearson":
                    _, p = sp_stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
                elif method == "spearman":
                    _, p = sp_stats.spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
                else:
                    _, p = sp_stats.kendalltau(valid.iloc[:, 0], valid.iloc[:, 1])
            else:
                p = np.nan
            star = _stars(p) if show_stars else ""
            result.iloc[i, j] = f"{_fmt_num(r_val, digits)}{star}"

    fn = FootnoteRegistry()
    if show_stars:
        fn.add("* p<0.05, ** p<0.01, *** p<0.001")
    fn.add(f"Method: {method.title()}")
    return _to_format(result, output_format, title=title, footnotes=fn.render(output_format))


# ---------------------------------------------------------------------------
# Model comparison table
# ---------------------------------------------------------------------------


def model_comparison_table(
    models: dict[str, Any],
    *,
    nested: bool = False,
    digits: int = 3,
    output_format: FormatTarget = "dataframe",
    title: str = "Model Comparison",
) -> pd.DataFrame | str:
    """
    Compare multiple fitted models on key fit statistics.

    Parameters
    ----------
    models : dict
        ``{model_name: fitted_model}``.  Models should be statsmodels
        result objects.
    nested : bool
        If ``True``, perform likelihood-ratio tests between successive
        models (assumes they are nested).
    digits : int
    output_format : FormatTarget
    title : str

    Returns
    -------
    DataFrame or str
    """
    records = []
    prev_llf = None
    prev_df = None
    prev_name = None

    for mname, model_obj in models.items():
        nobs = getattr(model_obj, "nobs", np.nan)
        df_model = getattr(model_obj, "df_model", np.nan)
        llf = getattr(model_obj, "llf", np.nan)
        aic = getattr(model_obj, "aic", np.nan)
        bic = getattr(model_obj, "bic", np.nan)
        r2 = getattr(model_obj, "rsquared", None)
        pseudo_r2 = getattr(model_obj, "prsquared", None)

        rec: dict[str, str] = {
            "Model": mname,
            "N": str(int(nobs)) if np.isfinite(nobs) else "",
            "df": _fmt_num(df_model, 0) if np.isfinite(df_model) else "",
            "Log-Lik": _fmt_num(llf, digits) if np.isfinite(llf) else "",
            "AIC": _fmt_num(aic, digits) if np.isfinite(aic) else "",
            "BIC": _fmt_num(bic, digits) if np.isfinite(bic) else "",
        }

        if r2 is not None:
            rec["R-sq"] = _fmt_num(r2, digits)
        elif pseudo_r2 is not None:
            rec["Pseudo R-sq"] = _fmt_num(pseudo_r2, digits)

        # LR test against previous model
        if nested and prev_llf is not None and np.isfinite(llf) and np.isfinite(prev_llf):
            lr_stat = 2 * (llf - prev_llf)
            delta_df = df_model - prev_df if prev_df is not None else 1
            if lr_stat > 0 and delta_df > 0:
                lr_p = sp_stats.chi2.sf(lr_stat, delta_df)
                rec["LR stat"] = _fmt_num(lr_stat, digits)
                rec["LR p"] = _fmt_pval(lr_p, 3)
            else:
                rec["LR stat"] = ""
                rec["LR p"] = ""
        elif nested:
            rec["LR stat"] = ""
            rec["LR p"] = ""

        prev_llf = llf
        prev_df = df_model
        prev_name = mname
        records.append(rec)

    result_df = pd.DataFrame(records).set_index("Model")
    fn = FootnoteRegistry()
    if nested:
        fn.add("LR test compares each model to the one above it")
    return _to_format(result_df, output_format, title=title, footnotes=fn.render(output_format))


# ---------------------------------------------------------------------------
# ANOVA table
# ---------------------------------------------------------------------------


def anova_table(
    model: Any,
    *,
    typ: int = 2,
    digits: int = 3,
    output_format: FormatTarget = "dataframe",
    title: str = "ANOVA Table",
) -> pd.DataFrame | str:
    """
    ANOVA table from a fitted OLS model.

    Parameters
    ----------
    model : statsmodels OLS results
        A fitted OLS model.
    typ : int
        ANOVA type (1, 2, or 3).
    digits : int
    output_format : FormatTarget
    title : str

    Returns
    -------
    DataFrame or str
    """
    import statsmodels.api as sm_api

    anova_df = sm_api.stats.anova_lm(model, typ=typ)

    # Format
    formatted = anova_df.copy()
    for col in ["sum_sq", "mean_sq", "F"]:
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(lambda x: _fmt_num(x, digits) if np.isfinite(x) else "")
    if "PR(>F)" in formatted.columns:
        formatted["p-value"] = formatted["PR(>F)"].apply(lambda x: _fmt_pval(x, 3) if np.isfinite(x) else "")
        formatted[""] = formatted["PR(>F)"].apply(lambda x: _stars(x) if np.isfinite(x) else "")
        formatted = formatted.drop(columns=["PR(>F)"])
    if "df" in formatted.columns:
        formatted["df"] = formatted["df"].apply(lambda x: str(int(x)) if np.isfinite(x) else "")

    fn = FootnoteRegistry()
    fn.add(f"Type {typ} ANOVA. * p<0.05, ** p<0.01, *** p<0.001")
    return _to_format(formatted, output_format, title=title, footnotes=fn.render(output_format))


# ---------------------------------------------------------------------------
# Custom number formatting
# ---------------------------------------------------------------------------


def format_number(
    x: float,
    *,
    style: str = "fixed",
    digits: int = 2,
    apa: bool = False,
) -> str:
    """
    Format a single number according to style conventions.

    Parameters
    ----------
    x : float
        The number to format.
    style : str
        ``"fixed"``, ``"scientific"``, ``"percent"``, or ``"integer"``.
    digits : int
        Significant digits or decimal places.
    apa : bool
        APA-style (suppress leading zeros for correlations/p-values).

    Returns
    -------
    str
    """
    if not np.isfinite(x):
        return ""
    if style == "fixed":
        return _fmt_num(x, digits, apa)
    if style == "scientific":
        return f"{x:.{digits}e}"
    if style == "percent":
        return f"{x * 100:.{digits}f}%"
    if style == "integer":
        return str(int(round(x)))
    raise ValueError(f"Unknown style: {style}")


# ---------------------------------------------------------------------------
# Batch formatting helper
# ---------------------------------------------------------------------------


def format_dataframe(
    df: pd.DataFrame,
    *,
    numeric_fmt: str = ".2f",
    pval_cols: list[str] | None = None,
    bold_cols: list[str] | None = None,
    output_format: FormatTarget = "dataframe",
    title: str = "",
) -> pd.DataFrame | str:
    """
    Apply uniform formatting to a DataFrame for publication.

    Parameters
    ----------
    df : DataFrame
    numeric_fmt : str
        Format string for numeric columns.
    pval_cols : list of str or None
        Columns to format as p-values.
    bold_cols : list of str or None
        Columns where values meeting significance should be bolded (HTML/LaTeX).
    output_format : FormatTarget
    title : str

    Returns
    -------
    DataFrame or str
    """
    formatted = df.copy()

    # Format numeric columns
    for col in formatted.select_dtypes(include="number").columns:
        if pval_cols and col in pval_cols:
            formatted[col] = formatted[col].apply(lambda x: _fmt_pval(x, 3))
        else:
            formatted[col] = formatted[col].apply(lambda x: f"{x:{numeric_fmt}}" if np.isfinite(x) else "")

    return _to_format(formatted, output_format, title=title)


# ---------------------------------------------------------------------------
# Summary statistics table
# ---------------------------------------------------------------------------


def summary_statistics_table(
    data: pd.DataFrame,
    variables: list[str] | None = None,
    *,
    stats: list[str] | None = None,
    digits: int = 2,
    output_format: FormatTarget = "dataframe",
    title: str = "Summary Statistics",
) -> pd.DataFrame | str:
    """
    Descriptive statistics table for a set of variables.

    Parameters
    ----------
    data : DataFrame
    variables : list of str or None
        Variables to summarise.  Defaults to all numeric columns.
    stats : list of str or None
        Statistics to compute.  Defaults to ``["n", "mean", "sd",
        "median", "min", "max", "missing"]``.
    digits : int
    output_format : FormatTarget
    title : str

    Returns
    -------
    DataFrame or str
    """
    if variables is None:
        variables = data.select_dtypes(include="number").columns.tolist()
    if stats is None:
        stats = ["n", "mean", "sd", "median", "min", "max", "missing"]

    stat_funcs = {
        "n": lambda s: s.notna().sum(),
        "mean": lambda s: s.mean(),
        "sd": lambda s: s.std(),
        "median": lambda s: s.median(),
        "min": lambda s: s.min(),
        "max": lambda s: s.max(),
        "missing": lambda s: s.isna().sum(),
        "pct_missing": lambda s: 100 * s.isna().mean(),
        "q25": lambda s: s.quantile(0.25),
        "q75": lambda s: s.quantile(0.75),
        "iqr": lambda s: s.quantile(0.75) - s.quantile(0.25),
        "skewness": lambda s: s.skew(),
        "kurtosis": lambda s: s.kurtosis(),
    }

    records = []
    for var in variables:
        rec: dict[str, str] = {"Variable": var}
        for stat in stats:
            fn = stat_funcs.get(stat)
            if fn is not None:
                val = fn(data[var])
                if stat in ("n", "missing"):
                    rec[stat] = str(int(val))
                else:
                    rec[stat] = _fmt_num(float(val), digits)
            else:
                rec[stat] = ""
        records.append(rec)

    result_df = pd.DataFrame(records).set_index("Variable")
    return _to_format(result_df, output_format, title=title)


# ---------------------------------------------------------------------------
# Treatment effect summary table
# ---------------------------------------------------------------------------


def treatment_effect_table(
    estimators: dict[str, dict[str, float]],
    *,
    digits: int = 3,
    output_format: FormatTarget = "dataframe",
    title: str = "Treatment Effect Estimates",
) -> pd.DataFrame | str:
    """
    Summary table of causal effect estimates from multiple estimators.

    Parameters
    ----------
    estimators : dict
        ``{estimator_name: {"estimate": float, "se": float,
        "ci_lower": float, "ci_upper": float, "p_value": float}}``.
    digits : int
    output_format : FormatTarget
    title : str

    Returns
    -------
    DataFrame or str
    """
    records = []
    for name, vals in estimators.items():
        est = vals.get("estimate", np.nan)
        se = vals.get("se", np.nan)
        ci_lo = vals.get("ci_lower", np.nan)
        ci_hi = vals.get("ci_upper", np.nan)
        pv = vals.get("p_value", np.nan)
        records.append(
            {
                "Estimator": name,
                "Estimate": _fmt_num(est, digits),
                "SE": _fmt_num(se, digits),
                "95% CI": f"({_fmt_num(ci_lo, digits)}, {_fmt_num(ci_hi, digits)})",
                "p-value": _fmt_pval(pv, 3),
                "": _stars(pv),
            }
        )

    result_df = pd.DataFrame(records).set_index("Estimator")
    fn = FootnoteRegistry()
    fn.add("* p<0.05, ** p<0.01, *** p<0.001")
    return _to_format(result_df, output_format, title=title, footnotes=fn.render(output_format))
