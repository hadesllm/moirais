"""morie.explain — human-readable descriptions of module-output CSVs.

Backs ``morie explain <filename>`` and ``morie cheatsheet``.

The explanations here are written for someone who has just run
``morie run-module <name>`` for the first time and is staring at a
folder of 10-15 CSVs not knowing where to start.  They describe:

  * what question the file answers,
  * how to read each column,
  * which row / cell to look at first.

Adding a new module's output files means adding entries here.
"""

from __future__ import annotations

# Map filename (no path) → multi-line explanation.
# Keep each entry to a paragraph + a short table; the goal is "scan
# in 30 seconds, know what to do next", not a methodology paper.
_EXPLANATIONS: dict[str, str] = {
    # ─── power-design outputs ───────────────────────────────────────────
    "power_summary.csv": """
Question this file answers: "What were the design assumptions for my
power analysis, and what's the recommended sample size at a glance?"

It's a one-row summary.  Read across; columns include:
  - effect_size          The minimum effect you said you want to detect
  - alpha                False-positive rate (typically 0.05)
  - power                True-positive rate (typically 0.80)
  - n_recommended        Per-group sample size to hit (power, effect) at alpha

If you want detail beyond the summary row, see the companion files
listed in `power_two_proportion_gender.csv` (two-proportion grid) and
`power_one_proportion_grid.csv` (one-proportion grid).
""".strip(),

    "power_two_proportion_gender.csv": """
Question this file answers: "How many participants per gender group
do I need to detect an effect of size X?"

Read each row: pick the effect_size you want to detect; the row tells
you the per-group sample size needed.

Columns:
  - effect_size          The difference in proportions you want to detect
                         (e.g. 0.05 = a 5 percentage-point gap between
                          men and women)
  - n_per_group          Number of participants per group required
  - power                Achieved power at this n (should match your design)
  - assumed_baseline     The baseline proportion the calculation uses

Typical usage: scroll to the row matching your hypothesised effect, read
n_per_group, double it for total sample.
""".strip(),

    "power_one_proportion_grid.csv": """
Same idea as power_two_proportion_gender.csv, but for a single-proportion
design (one group, no comparison): "What's the smallest deviation from
a null proportion p0 I can detect with n participants at alpha=0.05?"

Pick a row by n; read effect_size for the smallest detectable difference.
""".strip(),

    "power_ebac_endpoint_anchors.csv": """
Power-analysis grid specific to alcohol-impairment endpoints (eBAC =
estimated blood alcohol concentration).  Each row is one anchor point:
"if your endpoint is ebac_tot > 0.08 vs <= 0.08, what's the sample size?"
""".strip(),

    "power_gpower_reference_two_group.csv": """
Cross-reference: matches morie's two-group power numbers to G*Power
(the gold-standard reference tool).  If you're submitting to a journal
that demands G*Power, this is the table to cite.
""".strip(),

    "power_interaction_assumptions.csv": """
Assumptions used by the interaction-effect (e.g. gender × age) power
calculation.  Read this if you want to know what the interaction model
assumed before trusting power_interaction_pairwise_details.csv.
""".strip(),

    "power_interaction_feasibility_flags.csv": """
Flags for whether each proposed interaction cell is feasible at the
target sample size.  TRUE = enough data expected, FALSE = under-powered.
""".strip(),

    "power_interaction_group_allocations.csv": """
How the total sample is split across interaction cells (e.g. men 18-24,
men 25-44, women 18-24, women 25-44).  Tells you the per-cell n.
""".strip(),

    "power_interaction_imbalance_penalty.csv": """
The penalty to power introduced by unequal cell sizes.  If allocations
in power_interaction_group_allocations are skewed, this quantifies how
much power you lose vs. a balanced design.
""".strip(),

    "power_interaction_pairwise_details.csv": """
The detail backing power_interaction_assumptions.  Pairwise effect sizes
for each combination of interaction levels.
""".strip(),

    "power_interaction_sample_size_targets.csv": """
The sample-size *targets* (per cell) to hit your desired power for each
interaction comparison.  Compare to your actual allocations file.
""".strip(),

    "randomization_block_blueprints.csv": """
Pre-baked randomization-scheme blueprints (block sizes, stratification
factors).  Pick one and the *_example CSVs show what the resulting
allocation looks like.
""".strip(),

    "randomization_schedule_example_heavy_drinking_30d.csv": """
A *worked-example* randomization schedule using heavy-drinking-30-day as
the stratifying outcome.  Shows the participant id → arm assignment
table; useful for replicating in your own survey software.
""".strip(),

    "randomization_schedule_example_ebac_legal.csv": """
Worked-example randomization schedule stratified by ebac_legal (the
legal-limit blood-alcohol-concentration endpoint).
""".strip(),

    "randomization_schedule_example_ebac_tot.csv": """
Worked-example randomization schedule stratified by ebac_tot (the
total-impairment blood-alcohol-concentration endpoint).
""".strip(),

    # ─── data-wrangling outputs ─────────────────────────────────────────
    "data_na_summary.csv": """
Per-column missingness summary.  Each row is one input column; columns
include n_missing, pct_missing.  Read this BEFORE running any inference
module — fields with high missingness need imputation or exclusion.
""".strip(),

    "data_wrangling_log.csv": """
Step-by-step log of what the data-wrangling module did to your input
(renames, coercions, dropped rows).  Useful for the methods section.
""".strip(),

    # ─── descriptive-statistics outputs ─────────────────────────────────
    "binomial_summaries.csv": """
Survey-weighted binomial summaries (e.g. heavy_drinking_30d prevalence)
WITHOUT survey weights.  Compare against binomial_summaries_survey_weighted
to see how much the weights shift the estimates.
""".strip(),

    "binomial_summaries_survey_weighted.csv": """
Survey-weighted binomial summaries WITH the CPADS weighting variable
applied.  These are the prevalence estimates you'd report in a paper.
""".strip(),

    "probability_estimates.csv": """
Joint and conditional probability estimates across the survey design.
Read column by column; row labels indicate the conditioning event.
""".strip(),

    # ─── frequentist-inference outputs ──────────────────────────────────
    "frequentist_heavy_drinking_prevalence_ci.csv": """
Frequentist (Wilson / Clopper-Pearson) confidence intervals for the
prevalence of heavy drinking.  Each row is one subgroup; columns are
estimate, ci_lower, ci_upper.
""".strip(),

    "frequentist_effect_sizes.csv": """
Cohen's-d / odds-ratio / risk-difference effect sizes for the primary
contrasts of the analysis.  Read alongside p-values from
frequentist_hypothesis_tests.csv.
""".strip(),

    "frequentist_hypothesis_tests.csv": """
Per-contrast p-values and test statistics.  CAUTION: these are
NOT corrected for multiple comparisons by default — apply
Bonferroni / Benjamini-Hochberg yourself if your design demands it.
""".strip(),
}


def describe(filename: str) -> str:
    """Return the human-readable description of an output CSV, or a fallback."""
    # Normalise: strip path, lowercase, common suffix variations
    name = filename.rsplit("/", 1)[-1]
    if name in _EXPLANATIONS:
        return _EXPLANATIONS[name]
    # Try without extension swap
    base = name.rsplit(".", 1)[0]
    for candidate, body in _EXPLANATIONS.items():
        if candidate.rsplit(".", 1)[0] == base:
            return body
    return (
        f"No registered explanation for {name!r}.\n\n"
        f"Known files:\n"
        + "\n".join(f"  - {k}" for k in sorted(_EXPLANATIONS))
        + "\n\nIf you think this file should be explained, file an issue at "
        "https://github.com/hadesllm/morie/issues."
    )


# ─── cheatsheet ───────────────────────────────────────────────────────────

def _cheatsheet_body() -> str:
    """Build the cheatsheet with section headings translated via i18n.

    Commands stay English (they're literal CLI tokens), but section
    headings respect MORIE_LOCALE so a French / Spanish / Chinese /
    etc user actually sees their language on screen.
    """
    from .i18n import t
    return f"""
morie cheat sheet
=================

{t("cheatsheet.install")}
  curl -fsSL https://hadesllm.github.io/morie/install.sh | bash
  brew tap hadesllm/morie && brew install morie
  pip install morie
  docker run --rm ghcr.io/hadesllm/morie:latest morie --help

{t("cheatsheet.learn")}
  morie tutorial                  Interactive walkthrough
  morie cheatsheet                This card
  morie list-modules              List all 23 analysis modules
  morie list-datasets             List built-in datasets
  morie explain power_summary.csv What does this output mean?

{t("cheatsheet.run")}
  morie run-module power-design --output-dir out/
  morie run-module descriptive-statistics --output-dir out/
  morie run-module frequentist-inference --output-dir out/
  morie run-modules all --output-dir out/

{t("cheatsheet.pull")}
  morie pull tps-major --year 2024 --out tps-2024.csv
  morie pull tps-shootings --year 2024
  morie pull tps-homicide --year 2024
  morie pull tps-layers                                   # registry
  morie pull cpads --out cpads.csv                        # synth or real
  morie pull otis-a01-toy --out otis.csv                  # toy
  morie pull siu-toy --out siu.csv                        # toy SIU report

{t("cheatsheet.ingest")}
  morie ingest tps --layer major-crime --year 2024 --out tps.csv
  morie ingest ckan --portal https://open.canada.ca/data --search alcohol
  morie ingest siu --report-id 22-OFD-001 --out report/

{t("cheatsheet.help")}
  morie ask "I have a treatment-control design; what module fits?"
  morie doctor                    Check what's installed and working
  morie --help                    Top-level help

{t("cheatsheet.refs")}
  Docs:     https://hadesllm.github.io/morie/
  Issues:   https://github.com/hadesllm/morie/issues
  PyPI:     https://pypi.org/project/morie/
  R:        https://hadesllm.r-universe.dev/morie
""".strip()


# Static EN fallback kept for tests that import the literal string.
CHEATSHEET = _cheatsheet_body.__doc__ or ""


def print_cheatsheet() -> None:
    print(_cheatsheet_body())
