"""Automated report generation for epidemiological analyses.

Produces publication-ready statistical reports from MORIE pipeline outputs.
Supports APA-style statistical formatting, STROBE/CONSORT compliance
checking, multi-format output (Markdown, LaTeX, HTML), and automated
section generation for methods, results, limitations, and appendices.

The report generation system follows reporting guidelines endorsed by
EQUATOR Network for observational studies (STROBE), clinical trials
(CONSORT), and systematic reviews (PRISMA).

References
----------
Vandenbroucke, J. P. et al. (2007). Strengthening the Reporting of
Observational Studies in Epidemiology (STROBE): Explanation and elaboration.
*Annals of Internal Medicine*, 147(8), W163--W194.
https://doi.org/10.7326/0003-4819-147-8-200710160-00010-w1

Schulz, K. F. et al. (2010). CONSORT 2010 Statement: Updated guidelines
for reporting parallel group randomized trials. *BMJ*, 340, c332.
https://doi.org/10.1136/bmj.c332

American Psychological Association. (2020). *Publication Manual of the
American Psychological Association* (7th ed.).
"""

from __future__ import annotations

import logging
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ReportSection:
    """A single section of a report."""

    title: str
    content: str
    level: int = 2
    subsections: list[ReportSection] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render section to Markdown."""
        prefix = "#" * self.level
        parts = [f"{prefix} {self.title}", "", self.content, ""]
        for sub in self.subsections:
            parts.append(sub.to_markdown())
        return "\n".join(parts)

    def to_latex(self) -> str:
        """Render section to LaTeX."""
        levels = {1: "section", 2: "subsection", 3: "subsubsection", 4: "paragraph"}
        cmd = levels.get(self.level, "paragraph")
        parts = [f"\\{cmd}{{{self.title}}}", "", self.content, ""]
        for sub in self.subsections:
            parts.append(sub.to_latex())
        return "\n".join(parts)

    def to_html(self) -> str:
        """Render section to HTML."""
        tag = f"h{min(self.level, 6)}"
        # Simple Markdown-like conversion for body
        body = self.content.replace("\n\n", "</p><p>").replace("\n", " ")
        parts = [f"<{tag}>{self.title}</{tag}>", f"<p>{body}</p>"]
        for sub in self.subsections:
            parts.append(sub.to_html())
        return "\n".join(parts)


@dataclass
class ReportFigure:
    """A figure reference in a report."""

    number: int
    caption: str
    path: str
    label: str = ""

    def to_markdown(self) -> str:
        """Render figure reference in Markdown."""
        label_str = f" {{#fig-{self.label}}}" if self.label else ""
        return f"![Figure {self.number}. {self.caption}]({self.path}){label_str}"

    def to_latex(self) -> str:
        """Render figure in LaTeX."""
        label_str = f"\\label{{fig:{self.label}}}" if self.label else ""
        return textwrap.dedent(f"""\
            \\begin{{figure}}[htbp]
            \\centering
            \\includegraphics[width=0.8\\textwidth]{{{self.path}}}
            \\caption{{Figure {self.number}. {self.caption}}}
            {label_str}
            \\end{{figure}}""")


@dataclass
class ReportTable:
    """A table reference in a report."""

    number: int
    caption: str
    df: pd.DataFrame
    label: str = ""

    def to_markdown(self) -> str:
        """Render table in Markdown."""
        header = f"\n**Table {self.number}.** {self.caption}\n\n"
        return header + self.df.to_markdown(index=False) + "\n"

    def to_latex(self) -> str:
        """Render table in LaTeX."""
        label_str = f"\\label{{tab:{self.label}}}" if self.label else ""
        body = self.df.to_latex(index=False, escape=True)
        caption = f"\\caption{{Table {self.number}. {self.caption}}}"
        return f"\\begin{{table}}[htbp]\n\\centering\n{caption}\n{label_str}\n{body}\\end{{table}}\n"


@dataclass
class Report:
    """A complete report with sections, figures, and tables."""

    title: str
    authors: list[str]
    date: str
    sections: list[ReportSection] = field(default_factory=list)
    figures: list[ReportFigure] = field(default_factory=list)
    tables: list[ReportTable] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_section(self, section: ReportSection) -> None:
        """Add a section to the report."""
        self.sections.append(section)

    def add_figure(self, caption: str, path: str, label: str = "") -> ReportFigure:
        """Add a figure and return its reference."""
        fig = ReportFigure(
            number=len(self.figures) + 1,
            caption=caption,
            path=path,
            label=label or f"fig{len(self.figures) + 1}",
        )
        self.figures.append(fig)
        return fig

    def add_table(self, caption: str, df: pd.DataFrame, label: str = "") -> ReportTable:
        """Add a table and return its reference."""
        tbl = ReportTable(
            number=len(self.tables) + 1,
            caption=caption,
            df=df,
            label=label or f"tab{len(self.tables) + 1}",
        )
        self.tables.append(tbl)
        return tbl


@dataclass
class STROBECheck:
    """A single STROBE checklist item."""

    item_number: int
    category: str
    description: str
    present: bool
    evidence: str = ""
    recommendation: str = ""


@dataclass
class AuditResult:
    """Result of a statistical reporting audit."""

    file_path: str
    checks: list[dict[str, Any]]
    missing_elements: list[str]
    score: float  # 0-1 completeness score

    @property
    def complete(self) -> bool:
        """True if all required reporting elements are present."""
        return self.score >= 1.0


# ---------------------------------------------------------------------------
# APA-style statistical formatting
# ---------------------------------------------------------------------------


def format_p_value(p: float, *, threshold: float = 0.001) -> str:
    """Format a p-value in APA style.

    Parameters
    ----------
    p : float
        The p-value.
    threshold : float
        Below this threshold, report as ``p < threshold``.

    Returns
    -------
    str
        Formatted p-value string.

    Examples
    --------
    >>> format_p_value(0.034)
    'p = .034'
    >>> format_p_value(0.0002)
    'p < .001'
    >>> format_p_value(0.5)
    'p = .500'
    """
    if not np.isfinite(p):
        return "p = NaN"
    if p < 0:
        return "p < .001"
    if p < threshold:
        return f"p < {threshold:.3f}".replace("0.", ".")
    return f"p = {p:.3f}".replace("0.", ".")


def format_t_test(
    t: float,
    df: float,
    p: float,
    *,
    d: float | None = None,
) -> str:
    """Format a t-test result in APA style.

    Parameters
    ----------
    t : float
        t-statistic.
    df : float
        Degrees of freedom.
    p : float
        p-value.
    d : float | None
        Cohen's d effect size.

    Returns
    -------
    str
        APA-formatted string, e.g. ``t(98) = 2.45, p = .016, d = 0.49``.

    Examples
    --------
    >>> format_t_test(2.45, 98, 0.016, d=0.49)
    't(98) = 2.45, p = .016, d = 0.49'
    """
    parts = [f"t({df:.0f}) = {t:.2f}", format_p_value(p)]
    if d is not None:
        parts.append(f"d = {d:.2f}")
    return ", ".join(parts)


def format_f_test(
    f: float,
    df1: float,
    df2: float,
    p: float,
    *,
    eta_sq: float | None = None,
    partial_eta_sq: float | None = None,
) -> str:
    """Format an F-test result in APA style.

    Parameters
    ----------
    f : float
        F-statistic.
    df1 : float
        Numerator degrees of freedom.
    df2 : float
        Denominator degrees of freedom.
    p : float
        p-value.
    eta_sq : float | None
        Eta-squared effect size.
    partial_eta_sq : float | None
        Partial eta-squared.

    Returns
    -------
    str
        APA-formatted F-test string.

    Examples
    --------
    >>> format_f_test(4.56, 2, 97, 0.013, partial_eta_sq=0.086)
    'F(2, 97) = 4.56, p = .013, partial eta-sq = .086'
    """
    parts = [f"F({df1:.0f}, {df2:.0f}) = {f:.2f}", format_p_value(p)]
    if eta_sq is not None:
        parts.append(f"eta-sq = {eta_sq:.3f}".replace("0.", "."))
    if partial_eta_sq is not None:
        parts.append(f"partial eta-sq = {partial_eta_sq:.3f}".replace("0.", "."))
    return ", ".join(parts)


def format_chi_square(
    chi2: float,
    df: int,
    p: float,
    *,
    n: int | None = None,
    cramers_v: float | None = None,
) -> str:
    """Format a chi-square test result in APA style.

    Parameters
    ----------
    chi2 : float
        Chi-square statistic.
    df : int
        Degrees of freedom.
    p : float
        p-value.
    n : int | None
        Sample size.
    cramers_v : float | None
        Cramer's V effect size.

    Returns
    -------
    str
        APA-formatted chi-square string.

    Examples
    --------
    >>> format_chi_square(12.34, 3, 0.006, n=200, cramers_v=0.18)
    "chi-sq(3, N = 200) = 12.34, p = .006, V = 0.18"
    """
    if n is not None:
        stat = f"chi-sq({df}, N = {n}) = {chi2:.2f}"
    else:
        stat = f"chi-sq({df}) = {chi2:.2f}"
    parts = [stat, format_p_value(p)]
    if cramers_v is not None:
        parts.append(f"V = {cramers_v:.2f}")
    return ", ".join(parts)


def format_correlation(
    r: float,
    p: float,
    *,
    n: int | None = None,
    method: str = "Pearson",
) -> str:
    """Format a correlation result in APA style.

    Parameters
    ----------
    r : float
        Correlation coefficient.
    p : float
        p-value.
    n : int | None
        Sample size.
    method : str
        Correlation method name.

    Returns
    -------
    str
        APA-formatted correlation string.

    Examples
    --------
    >>> format_correlation(0.45, 0.002, n=50)
    'r(48) = .45, p = .002'
    """
    if n is not None:
        df_val = n - 2
        stat = f"r({df_val}) = {r:.2f}".replace("0.", ".", 1)
    else:
        stat = f"r = {r:.2f}".replace("0.", ".", 1)
    return f"{stat}, {format_p_value(p)}"


def format_odds_ratio(
    or_val: float,
    ci_lower: float,
    ci_upper: float,
    *,
    p: float | None = None,
) -> str:
    """Format an odds ratio with confidence interval.

    Parameters
    ----------
    or_val : float
        Odds ratio.
    ci_lower : float
        Lower 95% CI bound.
    ci_upper : float
        Upper 95% CI bound.
    p : float | None
        p-value.

    Returns
    -------
    str
        Formatted odds ratio string.

    Examples
    --------
    >>> format_odds_ratio(2.15, 1.34, 3.45, p=0.001)
    'OR = 2.15, 95% CI [1.34, 3.45], p = .001'
    """
    parts = [f"OR = {or_val:.2f}", f"95% CI [{ci_lower:.2f}, {ci_upper:.2f}]"]
    if p is not None:
        parts.append(format_p_value(p))
    return ", ".join(parts)


def format_hazard_ratio(
    hr: float,
    ci_lower: float,
    ci_upper: float,
    *,
    p: float | None = None,
) -> str:
    """Format a hazard ratio with confidence interval.

    Parameters
    ----------
    hr : float
        Hazard ratio.
    ci_lower : float
        Lower 95% CI bound.
    ci_upper : float
        Upper 95% CI bound.
    p : float | None
        p-value.

    Returns
    -------
    str

    Examples
    --------
    >>> format_hazard_ratio(1.85, 1.12, 3.05, p=0.016)
    'HR = 1.85, 95% CI [1.12, 3.05], p = .016'
    """
    parts = [f"HR = {hr:.2f}", f"95% CI [{ci_lower:.2f}, {ci_upper:.2f}]"]
    if p is not None:
        parts.append(format_p_value(p))
    return ", ".join(parts)


def format_mean_sd(
    mean: float,
    sd: float,
    *,
    n: int | None = None,
) -> str:
    """Format a mean and standard deviation.

    Parameters
    ----------
    mean : float
        Mean value.
    sd : float
        Standard deviation.
    n : int | None
        Sample size.

    Returns
    -------
    str
        Formatted string, e.g. ``M = 3.45, SD = 1.23``.

    Examples
    --------
    >>> format_mean_sd(3.45, 1.23, n=50)
    'M = 3.45, SD = 1.23 (n = 50)'
    """
    result = f"M = {mean:.2f}, SD = {sd:.2f}"
    if n is not None:
        result += f" (n = {n})"
    return result


def format_ci(
    estimate: float,
    ci_lower: float,
    ci_upper: float,
    *,
    level: float = 0.95,
    label: str = "",
) -> str:
    """Format an estimate with confidence interval.

    Parameters
    ----------
    estimate : float
        Point estimate.
    ci_lower : float
        Lower CI bound.
    ci_upper : float
        Upper CI bound.
    level : float
        Confidence level (default 0.95).
    label : str
        Optional label for the estimate.

    Returns
    -------
    str

    Examples
    --------
    >>> format_ci(0.35, 0.22, 0.48, label="ATE")
    'ATE = 0.35, 95% CI [0.22, 0.48]'
    """
    pct = int(level * 100)
    prefix = f"{label} = " if label else ""
    return f"{prefix}{estimate:.2f}, {pct}% CI [{ci_lower:.2f}, {ci_upper:.2f}]"


def format_regression_coefficient(
    name: str,
    beta: float,
    se: float,
    p: float,
    *,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
) -> str:
    """Format a regression coefficient.

    Parameters
    ----------
    name : str
        Variable name.
    beta : float
        Coefficient estimate.
    se : float
        Standard error.
    p : float
        p-value.
    ci_lower : float | None
        Lower CI bound.
    ci_upper : float | None
        Upper CI bound.

    Returns
    -------
    str

    Examples
    --------
    >>> format_regression_coefficient("age", 0.05, 0.02, 0.012, ci_lower=0.01, ci_upper=0.09)
    'age: B = 0.05, SE = 0.02, 95% CI [0.01, 0.09], p = .012'
    """
    parts = [f"{name}: B = {beta:.2f}", f"SE = {se:.2f}"]
    if ci_lower is not None and ci_upper is not None:
        parts.append(f"95% CI [{ci_lower:.2f}, {ci_upper:.2f}]")
    parts.append(format_p_value(p))
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------


def generate_introduction(
    *,
    topic: str,
    background: str = "",
    objectives: list[str] | None = None,
    hypotheses: list[str] | None = None,
) -> ReportSection:
    """Generate an introduction section.

    Parameters
    ----------
    topic : str
        Main research topic.
    background : str
        Background text.
    objectives : list[str] | None
        Study objectives.
    hypotheses : list[str] | None
        Study hypotheses.

    Returns
    -------
    ReportSection
    """
    parts = []
    if background:
        parts.append(background)
    else:
        parts.append(f"This study investigates {topic}.")

    subsections = []
    if objectives:
        obj_text = "\n".join(f"{i + 1}. {obj}" for i, obj in enumerate(objectives))
        subsections.append(
            ReportSection(
                title="Objectives",
                content=obj_text,
                level=3,
            )
        )

    if hypotheses:
        hyp_text = "\n".join(f"- **H{i + 1}**: {h}" for i, h in enumerate(hypotheses))
        subsections.append(
            ReportSection(
                title="Hypotheses",
                content=hyp_text,
                level=3,
            )
        )

    return ReportSection(
        title="Introduction",
        content="\n\n".join(parts),
        level=2,
        subsections=subsections,
    )


def generate_methods(
    *,
    study_design: str = "cross-sectional observational study",
    data_source: str = "",
    sample_description: str = "",
    variables: dict[str, str] | None = None,
    statistical_methods: list[str] | None = None,
    software: list[str] | None = None,
    alpha: float = 0.05,
) -> ReportSection:
    """Generate a methods section.

    Automatically describes the statistical methods used based on the
    parameters provided.

    Parameters
    ----------
    study_design : str
        Study design description.
    data_source : str
        Data source description.
    sample_description : str
        Sample/population description.
    variables : dict[str, str] | None
        Variable names mapped to descriptions (e.g. ``{"treatment": "cannabis use"}``).
    statistical_methods : list[str] | None
        Names of statistical methods used.
    software : list[str] | None
        Software/packages used.
    alpha : float
        Significance level.

    Returns
    -------
    ReportSection
    """
    subsections: list[ReportSection] = []

    # Study design
    design_text = f"This {study_design} was conducted to examine the research questions."
    if data_source:
        design_text += f" Data were obtained from {data_source}."
    subsections.append(
        ReportSection(
            title="Study Design",
            content=design_text,
            level=3,
        )
    )

    # Participants / Sample
    if sample_description:
        subsections.append(
            ReportSection(
                title="Participants",
                content=sample_description,
                level=3,
            )
        )

    # Variables
    if variables:
        var_lines = ["The following variables were included in the analysis:", ""]
        for var_name, var_desc in variables.items():
            var_lines.append(f"- **{var_name}**: {var_desc}")
        subsections.append(
            ReportSection(
                title="Variables",
                content="\n".join(var_lines),
                level=3,
            )
        )

    # Statistical Analysis
    stat_parts = []
    if statistical_methods:
        method_descriptions = _expand_method_descriptions(statistical_methods)
        stat_parts.extend(method_descriptions)
    stat_parts.append(
        f"All analyses were conducted at the {alpha} significance level (two-tailed) unless otherwise noted."
    )

    if software:
        sw_text = ", ".join(software)
        stat_parts.append(f"Analyses were performed using {sw_text}.")
    else:
        stat_parts.append(
            "Analyses were performed using the MORIE (Methods for Observational Inference and Robust Analysis of Interventions in Scientific Experimentation) Python package."
        )

    subsections.append(
        ReportSection(
            title="Statistical Analysis",
            content="\n\n".join(stat_parts),
            level=3,
        )
    )

    return ReportSection(
        title="Methods",
        content="",
        level=2,
        subsections=subsections,
    )


def _expand_method_descriptions(methods: list[str]) -> list[str]:
    """Map method names to brief statistical descriptions."""
    descriptions: dict[str, str] = {
        "ipw": (
            "Inverse probability weighting (IPW) was used to estimate causal "
            "effects while adjusting for confounding. Propensity scores were "
            "estimated using logistic regression."
        ),
        "aipw": (
            "Augmented inverse probability weighting (AIPW) was used, combining "
            "outcome modeling with propensity score weighting for doubly robust "
            "estimation (Robins, Rotnitzky, & Zhao, 1994)."
        ),
        "dml": (
            "Double machine learning (DML) was employed for semiparametric "
            "estimation with cross-fitting to avoid regularization bias "
            "(Chernozhukov et al., 2018)."
        ),
        "tmle": (
            "Targeted maximum likelihood estimation (TMLE) was used for "
            "efficient, doubly robust estimation of the target parameter "
            "(van der Laan & Rose, 2011)."
        ),
        "logistic": (
            "Multivariable logistic regression was used to estimate adjusted odds ratios with 95% confidence intervals."
        ),
        "linear": (
            "Multiple linear regression was used to estimate adjusted "
            "associations between exposures and the continuous outcome."
        ),
        "survey_weighted": (
            "All estimates incorporated survey sampling weights to account "
            "for the complex survey design. Variance estimation used "
            "Taylor series linearization."
        ),
        "propensity_matching": (
            "Propensity score matching was performed using nearest-neighbor "
            "matching with a caliper of 0.2 standard deviations of the logit "
            "of the propensity score (Austin, 2011)."
        ),
        "bootstrap": (
            "Bootstrap resampling (1000 iterations) was used for variance "
            "estimation and construction of confidence intervals."
        ),
        "sensitivity_analysis": (
            "Sensitivity analyses were conducted to assess the robustness of "
            "findings to alternative model specifications and unmeasured "
            "confounding."
        ),
    }

    expanded = []
    for method in methods:
        key = method.lower().replace(" ", "_").replace("-", "_")
        if key in descriptions:
            expanded.append(descriptions[key])
        else:
            expanded.append(f"{method} was used in the analysis.")
    return expanded


def generate_results(
    *,
    sample_size: int | None = None,
    descriptive_results: str | None = None,
    primary_results: str | None = None,
    secondary_results: str | None = None,
    sensitivity_results: str | None = None,
) -> ReportSection:
    """Generate a results section.

    Parameters
    ----------
    sample_size : int | None
        Total sample size.
    descriptive_results : str | None
        Text for descriptive statistics subsection.
    primary_results : str | None
        Primary analysis results.
    secondary_results : str | None
        Secondary analysis results.
    sensitivity_results : str | None
        Sensitivity analysis results.

    Returns
    -------
    ReportSection
    """
    subsections: list[ReportSection] = []

    if sample_size is not None or descriptive_results:
        desc = descriptive_results or ""
        if sample_size:
            desc = f"The final analytic sample comprised {sample_size:,} participants. " + desc
        subsections.append(
            ReportSection(
                title="Sample Description",
                content=desc,
                level=3,
            )
        )

    if primary_results:
        subsections.append(
            ReportSection(
                title="Primary Analysis",
                content=primary_results,
                level=3,
            )
        )

    if secondary_results:
        subsections.append(
            ReportSection(
                title="Secondary Analyses",
                content=secondary_results,
                level=3,
            )
        )

    if sensitivity_results:
        subsections.append(
            ReportSection(
                title="Sensitivity Analyses",
                content=sensitivity_results,
                level=3,
            )
        )

    return ReportSection(
        title="Results",
        content="",
        level=2,
        subsections=subsections,
    )


def generate_discussion(
    *,
    key_findings: list[str] | None = None,
    interpretation: str = "",
    comparison_to_literature: str = "",
    implications: str = "",
    future_directions: str = "",
) -> ReportSection:
    """Generate a discussion section.

    Parameters
    ----------
    key_findings : list[str] | None
        Key findings to summarize.
    interpretation : str
        Interpretation of results.
    comparison_to_literature : str
        Comparison to existing literature.
    implications : str
        Implications for practice or policy.
    future_directions : str
        Future research directions.

    Returns
    -------
    ReportSection
    """
    subsections: list[ReportSection] = []

    if key_findings:
        findings_text = "\n".join(f"- {f}" for f in key_findings)
        subsections.append(
            ReportSection(
                title="Key Findings",
                content=findings_text,
                level=3,
            )
        )

    if interpretation:
        subsections.append(
            ReportSection(
                title="Interpretation",
                content=interpretation,
                level=3,
            )
        )

    if comparison_to_literature:
        subsections.append(
            ReportSection(
                title="Comparison to Existing Literature",
                content=comparison_to_literature,
                level=3,
            )
        )

    if implications:
        subsections.append(
            ReportSection(
                title="Implications",
                content=implications,
                level=3,
            )
        )

    if future_directions:
        subsections.append(
            ReportSection(
                title="Future Directions",
                content=future_directions,
                level=3,
            )
        )

    return ReportSection(
        title="Discussion",
        content="",
        level=2,
        subsections=subsections,
    )


def generate_limitations(
    *,
    study_design: str = "cross-sectional",
    custom_limitations: list[str] | None = None,
) -> ReportSection:
    """Auto-generate a limitations section based on study design.

    Identifies common limitations for the given study design and appends
    any custom limitations provided.

    Parameters
    ----------
    study_design : str
        Study design type.
    custom_limitations : list[str] | None
        Additional limitations.

    Returns
    -------
    ReportSection
    """
    common: dict[str, list[str]] = {
        "cross-sectional": [
            "The cross-sectional design precludes causal inference, as "
            "temporality between exposure and outcome cannot be established.",
            "Self-reported data may be subject to recall bias and social desirability bias.",
        ],
        "cohort": [
            "Loss to follow-up may introduce selection bias if dropout is related to both exposure and outcome.",
            "Unmeasured confounding cannot be entirely ruled out despite adjustment for known confounders.",
        ],
        "case-control": [
            "Case-control studies are prone to recall bias, as cases may "
            "differentially recall past exposures compared to controls.",
            "Selection of appropriate controls is critical and may introduce selection bias.",
        ],
        "rct": [
            "Generalizability may be limited by strict eligibility criteria.",
            "Blinding was not possible for all participants and study staff.",
        ],
        "survey": [
            "Survey non-response may introduce bias if non-respondents differ systematically from respondents.",
            "Self-report measures may be subject to measurement error.",
            "Complex survey weights were used but may not fully account for all sources of non-representativeness.",
        ],
    }

    key = study_design.lower().replace("-", "_").replace(" ", "_")
    limitations = list(common.get(key, common["cross-sectional"]))

    # Universal limitations
    limitations.append("Residual confounding by unmeasured variables remains possible.")

    if custom_limitations:
        limitations.extend(custom_limitations)

    content = "\n\n".join(f"{i + 1}. {lim}" for i, lim in enumerate(limitations))

    return ReportSection(
        title="Limitations",
        content=content,
        level=3,
    )


def generate_executive_summary(
    *,
    background: str,
    methods_summary: str,
    key_findings: list[str],
    conclusions: str,
) -> ReportSection:
    """Generate an executive summary.

    Parameters
    ----------
    background : str
        Brief background.
    methods_summary : str
        Brief methods description.
    key_findings : list[str]
        Bullet points of key findings.
    conclusions : str
        Main conclusions.

    Returns
    -------
    ReportSection
    """
    findings_text = "\n".join(f"- {f}" for f in key_findings)

    content = textwrap.dedent(f"""\
        **Background**: {background}

        **Methods**: {methods_summary}

        **Key Findings**:
        {findings_text}

        **Conclusions**: {conclusions}
    """)

    return ReportSection(
        title="Executive Summary",
        content=content,
        level=2,
    )


def generate_appendix(
    *,
    supplementary_tables: list[tuple[str, pd.DataFrame]] | None = None,
    sensitivity_analyses: list[tuple[str, str]] | None = None,
    additional_methods: str | None = None,
) -> ReportSection:
    """Generate an appendix section.

    Parameters
    ----------
    supplementary_tables : list[tuple[str, pd.DataFrame]] | None
        List of (caption, DataFrame) tuples.
    sensitivity_analyses : list[tuple[str, str]] | None
        List of (title, content) tuples.
    additional_methods : str | None
        Additional methodological details.

    Returns
    -------
    ReportSection
    """
    subsections: list[ReportSection] = []

    if additional_methods:
        subsections.append(
            ReportSection(
                title="Additional Methodological Details",
                content=additional_methods,
                level=3,
            )
        )

    if supplementary_tables:
        table_parts = []
        for i, (caption, df) in enumerate(supplementary_tables, 1):
            table_parts.append(f"**Supplementary Table S{i}. {caption}**\n")
            table_parts.append(df.to_markdown(index=False))
            table_parts.append("")
        subsections.append(
            ReportSection(
                title="Supplementary Tables",
                content="\n".join(table_parts),
                level=3,
            )
        )

    if sensitivity_analyses:
        for sa_title, sa_content in sensitivity_analyses:
            subsections.append(
                ReportSection(
                    title=sa_title,
                    content=sa_content,
                    level=3,
                )
            )

    return ReportSection(
        title="Appendix",
        content="",
        level=2,
        subsections=subsections,
    )


# ---------------------------------------------------------------------------
# CONSORT flow diagram narrative
# ---------------------------------------------------------------------------


def generate_consort_narrative(
    *,
    assessed_for_eligibility: int,
    excluded: int,
    exclusion_reasons: dict[str, int] | None = None,
    randomized: int | None = None,
    allocated_treatment: int | None = None,
    allocated_control: int | None = None,
    lost_to_followup_treatment: int = 0,
    lost_to_followup_control: int = 0,
    analyzed_treatment: int | None = None,
    analyzed_control: int | None = None,
) -> ReportSection:
    """Generate a CONSORT-style flow diagram narrative.

    Parameters
    ----------
    assessed_for_eligibility : int
        Number assessed for eligibility.
    excluded : int
        Number excluded.
    exclusion_reasons : dict[str, int] | None
        Reasons for exclusion with counts.
    randomized : int | None
        Number randomized (or included for observational).
    allocated_treatment : int | None
        Number in treatment group.
    allocated_control : int | None
        Number in control group.
    lost_to_followup_treatment : int
        Lost to follow-up in treatment.
    lost_to_followup_control : int
        Lost to follow-up in control.
    analyzed_treatment : int | None
        Number analyzed in treatment.
    analyzed_control : int | None
        Number analyzed in control.

    Returns
    -------
    ReportSection
    """
    included = randomized or (assessed_for_eligibility - excluded)

    lines = [
        f"A total of {assessed_for_eligibility:,} individuals were assessed "
        f"for eligibility. Of these, {excluded:,} were excluded",
    ]

    if exclusion_reasons:
        reasons = [f"{reason} (n = {count:,})" for reason, count in exclusion_reasons.items()]
        lines[-1] += ": " + "; ".join(reasons) + "."
    else:
        lines[-1] += "."

    lines.append(f"The remaining {included:,} participants were included in the study.")

    if allocated_treatment is not None and allocated_control is not None:
        lines.append(
            f"Participants were allocated to the treatment group (n = {allocated_treatment:,}) "
            f"and the control group (n = {allocated_control:,})."
        )

    if lost_to_followup_treatment > 0 or lost_to_followup_control > 0:
        lines.append(
            f"During follow-up, {lost_to_followup_treatment:,} were lost from "
            f"the treatment group and {lost_to_followup_control:,} from the control group."
        )

    if analyzed_treatment is not None and analyzed_control is not None:
        lines.append(
            f"The final analysis included {analyzed_treatment:,} in the treatment group "
            f"and {analyzed_control:,} in the control group "
            f"(total N = {analyzed_treatment + analyzed_control:,})."
        )

    return ReportSection(
        title="Participant Flow",
        content="\n\n".join(lines),
        level=3,
    )


# ---------------------------------------------------------------------------
# STROBE checklist
# ---------------------------------------------------------------------------

_STROBE_ITEMS: list[tuple[int, str, str]] = [
    (
        1,
        "Title and abstract",
        "Indicate the study design with a commonly used term; provide an informative and balanced summary",
    ),
    (2, "Background/rationale", "Explain the scientific background and rationale"),
    (3, "Objectives", "State specific objectives, including any prespecified hypotheses"),
    (4, "Study design", "Present key elements of study design early in the paper"),
    (5, "Setting", "Describe the setting, locations, and relevant dates"),
    (6, "Participants", "Give the eligibility criteria, and the sources and methods of selection"),
    (7, "Variables", "Clearly define all outcomes, exposures, predictors, potential confounders, and effect modifiers"),
    (8, "Data sources/measurement", "Give sources of data and details of methods of assessment"),
    (9, "Bias", "Describe any efforts to address potential sources of bias"),
    (10, "Study size", "Explain how the study size was arrived at"),
    (11, "Quantitative variables", "Explain how quantitative variables were handled in the analyses"),
    (12, "Statistical methods", "Describe all statistical methods, including those used to control for confounding"),
    (13, "Participants", "Report numbers at each stage of study; give reasons for non-participation"),
    (14, "Descriptive data", "Give characteristics of study participants and information on exposures"),
    (15, "Outcome data", "Report numbers of outcome events or summary measures"),
    (16, "Main results", "Give unadjusted estimates and, if applicable, confounder-adjusted estimates"),
    (17, "Other analyses", "Report other analyses done (subgroup, interaction, sensitivity)"),
    (18, "Key results", "Summarise key results with reference to study objectives"),
    (
        19,
        "Limitations",
        "Discuss limitations of the study, taking into account sources of potential bias or imprecision",
    ),
    (20, "Interpretation", "Give a cautious overall interpretation of results"),
    (21, "Generalisability", "Discuss the generalisability of the study results"),
    (22, "Funding", "Give the source of funding and the role of funders"),
]


def check_strobe_compliance(
    report_text: str,
) -> list[STROBECheck]:
    """Check a report for STROBE checklist compliance.

    Uses keyword matching to detect the presence of required reporting
    elements.  This is a heuristic check, not a substitute for expert
    review.

    Parameters
    ----------
    report_text : str
        Full report text.

    Returns
    -------
    list[STROBECheck]
        One check per STROBE item.
    """
    text_lower = report_text.lower()
    results: list[STROBECheck] = []

    keyword_map: dict[int, list[str]] = {
        1: ["abstract", "summary"],
        2: ["background", "rationale", "introduction"],
        3: ["objective", "aim", "hypothesis"],
        4: ["study design", "cross-sectional", "cohort", "case-control", "randomized"],
        5: ["setting", "location", "period", "dates"],
        6: ["eligibility", "inclusion", "exclusion", "criteria", "participant"],
        7: ["outcome", "exposure", "covariate", "confounder", "variable"],
        8: ["data source", "measurement", "instrument", "questionnaire", "survey"],
        9: ["bias", "confounding", "selection bias", "information bias"],
        10: ["sample size", "power", "power analysis"],
        11: ["categoriz", "continuous", "cutpoint", "quartile"],
        12: ["statistical method", "regression", "logistic", "model", "analysis"],
        13: ["flow", "attrition", "non-response", "missing"],
        14: ["table 1", "baseline", "characteristic", "demographic"],
        15: ["prevalence", "incidence", "event", "outcome data"],
        16: ["unadjusted", "adjusted", "crude", "confounder-adjusted", "odds ratio", "coefficient"],
        17: ["subgroup", "interaction", "sensitivity", "supplementary"],
        18: ["key finding", "summary", "main result"],
        19: ["limitation", "weakness", "bias"],
        20: ["interpretation", "implication", "conclusion"],
        21: ["generali", "external validity", "applicab"],
        22: ["funding", "grant", "financial", "support"],
    }

    for item_num, category, description in _STROBE_ITEMS:
        keywords = keyword_map.get(item_num, [])
        found_keywords = [kw for kw in keywords if kw in text_lower]
        present = len(found_keywords) > 0

        evidence = f"Found: {', '.join(found_keywords)}" if found_keywords else "Not detected"
        recommendation = "" if present else f"Consider adding {category.lower()} information."

        results.append(
            STROBECheck(
                item_number=item_num,
                category=category,
                description=description,
                present=present,
                evidence=evidence,
                recommendation=recommendation,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Statistical reporting audit
# ---------------------------------------------------------------------------


def audit_statistical_reporting(
    path: str | Path,
) -> AuditResult:
    """Audit a CSV of statistical results for completeness.

    Checks whether required reporting elements are present:
    point estimates, standard errors, confidence intervals, p-values,
    sample sizes, and effect sizes.

    Parameters
    ----------
    path : str | Path
        Path to a CSV file with statistical results.

    Returns
    -------
    AuditResult
    """
    path = Path(path)
    if not path.is_file():
        return AuditResult(
            file_path=str(path),
            checks=[],
            missing_elements=["file not found"],
            score=0.0,
        )

    try:
        df = pd.read_csv(path)
    except Exception:
        return AuditResult(
            file_path=str(path),
            checks=[],
            missing_elements=["unreadable file"],
            score=0.0,
        )

    cols_lower = [c.lower() for c in df.columns]
    checks: list[dict[str, Any]] = []
    missing: list[str] = []

    required_patterns: list[tuple[str, str]] = [
        ("point_estimate", r"(estimate|coef|beta|effect|or|hr|rr|ate|att|mean)"),
        ("standard_error", r"(se|std.err|standard.error)"),
        ("confidence_interval", r"(ci|conf|lower|upper)"),
        ("p_value", r"(p.val|pvalue|^p$|significance)"),
        ("sample_size", r"(^n$|n.obs|sample.size|nobs)"),
    ]

    found = 0
    total = len(required_patterns)

    for element_name, pattern in required_patterns:
        pat = re.compile(pattern, re.IGNORECASE)
        matched = any(pat.search(c) for c in df.columns)
        checks.append(
            {
                "element": element_name,
                "present": matched,
                "columns": [c for c in df.columns if pat.search(c)],
            }
        )
        if matched:
            found += 1
        else:
            missing.append(element_name)

    score = found / total if total > 0 else 0.0

    return AuditResult(
        file_path=str(path),
        checks=checks,
        missing_elements=missing,
        score=score,
    )


# ---------------------------------------------------------------------------
# Boilerplate generators
# ---------------------------------------------------------------------------


def generate_reproducibility_statement(
    *,
    software: list[str] | None = None,
    seed: int | None = None,
    data_available: bool = False,
    code_available: bool = True,
    repository_url: str = "",
) -> str:
    """Generate a reproducibility statement.

    Parameters
    ----------
    software : list[str] | None
        Software and versions used.
    seed : int | None
        Random seed used.
    data_available : bool
        Whether data are publicly available.
    code_available : bool
        Whether analysis code is available.
    repository_url : str
        URL to code repository.

    Returns
    -------
    str
    """
    parts = ["**Reproducibility Statement**", ""]

    if software:
        parts.append("Analyses were conducted using: " + ", ".join(software) + ".")
    else:
        parts.append("Analyses were conducted using the MORIE Python package.")

    if seed is not None:
        parts.append(f"The random seed was set to {seed} for all stochastic procedures.")

    if data_available:
        parts.append("Data used in this analysis are publicly available.")
    else:
        parts.append(
            "Due to privacy restrictions, individual-level data cannot be "
            "shared publicly. Aggregate results and synthetic data are available "
            "upon request."
        )

    if code_available:
        if repository_url:
            parts.append(f"Analysis code is available at: {repository_url}")
        else:
            parts.append("Analysis code is available upon request.")

    return "\n".join(parts)


def generate_data_availability_statement(
    *,
    data_public: bool = False,
    repository: str = "",
    access_conditions: str = "",
    contact: str = "",
) -> str:
    """Generate a data availability statement.

    Parameters
    ----------
    data_public : bool
        Whether data are publicly available.
    repository : str
        Data repository URL.
    access_conditions : str
        Conditions for data access.
    contact : str
        Contact for data requests.

    Returns
    -------
    str
    """
    if data_public and repository:
        return (
            f"**Data Availability**: The data supporting the findings of this study "
            f"are publicly available at {repository}."
        )
    elif data_public:
        return "**Data Availability**: The data supporting the findings of this study are publicly available."
    else:
        text = (
            "**Data Availability**: The data that support the findings of this "
            "study are not publicly available due to privacy restrictions."
        )
        if access_conditions:
            text += f" {access_conditions}"
        if contact:
            text += f" Requests for data access should be directed to {contact}."
        return text


def generate_conflict_of_interest() -> str:
    """Generate a conflict of interest statement template.

    Returns
    -------
    str
    """
    return ("**Conflict of Interest**: The authors declare no conflict "
            "of interest.")


def generate_funding_acknowledgment(
    *,
    funders: list[dict[str, str]] | None = None,
) -> str:
    """Generate a funding acknowledgment section.

    Parameters
    ----------
    funders : list[dict[str, str]] | None
        List of dicts with keys ``name``, ``grant_number``.

    Returns
    -------
    str
    """
    if not funders:
        return "**Funding**: This research received no specific funding."

    parts = ["**Funding**: This research was supported by"]
    funder_strs = []
    for f in funders:
        name = f.get("name", "")
        grant = f.get("grant_number", "")
        if grant:
            funder_strs.append(f"{name} (grant {grant})")
        else:
            funder_strs.append(name)
    parts[0] += " " + "; ".join(funder_strs) + "."
    return parts[0]


def generate_author_contributions(
    contributions: dict[str, list[str]],
) -> str:
    """Generate an author contribution statement using CRediT taxonomy.

    Parameters
    ----------
    contributions : dict[str, list[str]]
        Mapping of author names to CRediT roles.
        Valid roles: Conceptualization, Methodology, Software, Validation,
        Formal analysis, Investigation, Resources, Data curation,
        Writing - original draft, Writing - review & editing,
        Visualization, Supervision, Project administration,
        Funding acquisition.

    Returns
    -------
    str

    Examples
    --------
    >>> generate_author_contributions({
    ...     "Smith, J.": ["Conceptualization", "Methodology", "Writing - original draft"],
    ...     "Doe, A.": ["Formal analysis", "Software", "Writing - review & editing"],
    ... })
    '**Author Contributions**:\n\nSmith, J.: Conceptualization, Methodology, Writing - original draft.\nDoe, A.: Formal analysis, Software, Writing - review & editing.'
    """
    lines = ["**Author Contributions**:", ""]
    for author, roles in contributions.items():
        lines.append(f"{author}: {', '.join(roles)}.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report compilation
# ---------------------------------------------------------------------------


def compile_report(
    report: Report,
    *,
    output_format: str = "markdown",
) -> str:
    """Compile a Report object into a single document string.

    Parameters
    ----------
    report : Report
        Report to compile.
    output_format : str
        ``markdown``, ``latex``, or ``html``.

    Returns
    -------
    str
        Compiled report text.
    """
    parts: list[str] = []

    if output_format == "markdown":
        parts.append(f"# {report.title}")
        parts.append("")
        parts.append(f"A journey of a thousand miles begins with a single step. -- Lao Tzu")
        parts.append(f"**Date**: {report.date}")
        parts.append("")
        for section in report.sections:
            parts.append(section.to_markdown())
        for fig in report.figures:
            parts.append(fig.to_markdown())
            parts.append("")
        for tbl in report.tables:
            parts.append(tbl.to_markdown())

    elif output_format == "latex":
        parts.append(f"\\title{{{report.title}}}")
        author_str = " \\and ".join(report.authors)
        parts.append(f"A journey of a thousand miles begins with a single step. -- Lao Tzu")
        parts.append(f"\\date{{{report.date}}}")
        parts.append("\\maketitle")
        parts.append("")
        for section in report.sections:
            parts.append(section.to_latex())
        for fig in report.figures:
            parts.append(fig.to_latex())
        for tbl in report.tables:
            parts.append(tbl.to_latex())

    elif output_format == "html":
        parts.append(f"<h1>{report.title}</h1>")
        parts.append(f"A journey of a thousand miles begins with a single step. -- Lao Tzu")
        parts.append(f"<p><strong>Date</strong>: {report.date}</p>")
        for section in report.sections:
            parts.append(section.to_html())

    return "\n".join(parts)


def save_report(
    report: Report,
    path: str | Path,
    *,
    output_format: str | None = None,
) -> Path:
    """Compile and save a report to a file.

    Parameters
    ----------
    report : Report
        Report to save.
    path : str | Path
        Output file path.
    output_format : str | None
        If None, inferred from file extension.

    Returns
    -------
    Path
        Path to the saved file.
    """
    path = Path(path)
    if output_format is None:
        ext_map = {".md": "markdown", ".tex": "latex", ".html": "html"}
        output_format = ext_map.get(path.suffix.lower(), "markdown")

    content = compile_report(report, output_format=output_format)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("Report saved to %s (%s format)", path, output_format)
    return path


def compare_reports(
    path_a: str | Path,
    path_b: str | Path,
) -> dict[str, Any]:
    """Compare two report versions.

    Parameters
    ----------
    path_a : str | Path
        First report.
    path_b : str | Path
        Second report.

    Returns
    -------
    dict[str, Any]
        Comparison with keys ``lines_added``, ``lines_removed``,
        ``sections_changed``.
    """
    text_a = Path(path_a).read_text(encoding="utf-8").splitlines()
    text_b = Path(path_b).read_text(encoding="utf-8").splitlines()

    set_a = set(text_a)
    set_b = set(text_b)

    added = set_b - set_a
    removed = set_a - set_b

    # Count section headings changed
    headings_a = {l for l in text_a if l.startswith("#")}
    headings_b = {l for l in text_b if l.startswith("#")}
    sections_added = headings_b - headings_a
    sections_removed = headings_a - headings_b

    return {
        "path_a": str(path_a),
        "path_b": str(path_b),
        "lines_a": len(text_a),
        "lines_b": len(text_b),
        "lines_added": len(added),
        "lines_removed": len(removed),
        "sections_added": list(sections_added),
        "sections_removed": list(sections_removed),
    }


# ---------------------------------------------------------------------------
# Full pipeline report generation
# ---------------------------------------------------------------------------


def generate_full_report(
    results_dir: str | Path,
    *,
    title: str = "MORIE Analysis Report",
    authors: list[str] | None = None,
    study_design: str = "cross-sectional observational study",
    output_path: str | Path | None = None,
    output_format: str = "markdown",
) -> Report:
    """Generate a complete report from pipeline output CSVs.

    Scans a results directory, detects statistical outputs, formats them
    into report sections, and compiles the full document.

    Parameters
    ----------
    results_dir : str | Path
        Directory containing CSV outputs from MORIE modules.
    title : str
        Report title.
    authors : list[str] | None
        Author names.
    study_design : str
        Study design for methods section.
    output_path : str | Path | None
        If provided, save the report to this path.
    output_format : str
        Output format (``markdown``, ``latex``, ``html``).

    Returns
    -------
    Report
        The compiled report object.
    """
    results_dir = Path(results_dir)
    if authors is None:
        authors = ["MORIE Analysis Pipeline"]

    report = Report(
        title=title,
        authors=authors,
        date=datetime.now().strftime("%Y-%m-%d"),
    )

    # Detect available results
    csv_files = sorted(results_dir.glob("*.csv")) if results_dir.is_dir() else []

    # Build methods based on detected outputs
    detected_methods: list[str] = []
    for csv_file in csv_files:
        name = csv_file.stem.lower()
        if "ipw" in name:
            detected_methods.append("ipw")
        if "dml" in name:
            detected_methods.append("dml")
        if "logistic" in name or "_or" in name:
            detected_methods.append("logistic")
        if "linear" in name or "coefficient" in name:
            detected_methods.append("linear")
        if "weighted" in name or "survey" in name:
            detected_methods.append("survey_weighted")
        if "bootstrap" in name:
            detected_methods.append("bootstrap")
        if "sensitivity" in name or "smote" in name:
            detected_methods.append("sensitivity_analysis")

    detected_methods = list(dict.fromkeys(detected_methods))  # deduplicate preserving order

    report.add_section(generate_introduction(topic=title))
    report.add_section(
        generate_methods(
            study_design=study_design,
            statistical_methods=detected_methods,
        )
    )

    # Build results from CSVs
    results_text_parts: list[str] = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                continue

            report.add_table(
                caption=csv_file.stem.replace("_", " ").title(),
                df=df.head(20),
                label=csv_file.stem,
            )
            results_text_parts.append(
                f"Results from {csv_file.stem.replace('_', ' ')} are presented in Table {len(report.tables)}."
            )
        except Exception as exc:
            logger.warning("Could not read %s: %s", csv_file, exc)

    report.add_section(
        generate_results(
            primary_results="\n\n".join(results_text_parts) if results_text_parts else "No results available.",
        )
    )

    report.add_section(generate_discussion())
    report.add_section(generate_limitations(study_design=study_design))

    if output_path:
        save_report(report, output_path, output_format=output_format)

    return report


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def render_strobe_checklist(checks: list[STROBECheck]) -> None:
    """Display STROBE compliance results with rich formatting.

    Parameters
    ----------
    checks : list[STROBECheck]
        STROBE check results.
    """
    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(
            title="STROBE Compliance Checklist",
            box=box.ROUNDED,
            header_style="bold cyan",
        )
        table.add_column("#", width=4, justify="right")
        table.add_column("Status", width=6, justify="center")
        table.add_column("Category", style="bold")
        table.add_column("Evidence")

        present_count = sum(1 for c in checks if c.present)

        for check in checks:
            status = "[green]PASS[/green]" if check.present else "[red]MISS[/red]"
            table.add_row(
                str(check.item_number),
                status,
                check.category,
                check.evidence,
            )

        console.print(table)
        console.print(
            f"\nCompliance: {present_count}/{len(checks)} items detected ({100 * present_count / len(checks):.0f}%)"
        )

    except ImportError:
        for check in checks:
            status = "PASS" if check.present else "MISS"
            print(f"  [{status}] {check.item_number}. {check.category}: {check.evidence}")


def render_audit_result(result: AuditResult) -> None:
    """Display statistical reporting audit with rich formatting.

    Parameters
    ----------
    result : AuditResult
        Audit result to display.
    """
    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(
            title=f"Statistical Reporting Audit: {Path(result.file_path).name}",
            box=box.ROUNDED,
            header_style="bold cyan",
        )
        table.add_column("Element", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Columns")

        for check in result.checks:
            status = "[green]FOUND[/green]" if check["present"] else "[red]MISSING[/red]"
            cols = ", ".join(check.get("columns", []))
            table.add_row(check["element"], status, cols)

        console.print(table)
        console.print(f"\nCompleteness score: {result.score:.0%}")

    except ImportError:
        for check in result.checks:
            status = "FOUND" if check["present"] else "MISSING"
            print(f"  [{status}] {check['element']}")
