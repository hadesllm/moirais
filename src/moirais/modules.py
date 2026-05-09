"""Module-level execution surface for MOIRAIS dataset analyses."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from .causal import run_ebac_selection_ipw_analysis, run_propensity_ipw_analysis
from .cpads import canonicalize_cpads_frame
from .data import DatasetRegistry
from .investigation import (
    compare_nested_logistic_models,
    run_treatment_effects_analysis,
    run_weighted_logistic_analysis,
)


def _find_cpads_csv() -> str:
    """Search for the CPADS PUMF microdata CSV relative to project root."""
    from .data import _project_root

    root = _project_root()
    candidate = root / "data" / "datasets" / "oc" / "CPADS" / "2021-2022" / "cpads-2021-2022-pumf2.csv"
    if candidate.exists():
        return str(candidate)
    return str(candidate)  # return path even if missing — error at load time


DEFAULT_CPADS_CSV = _find_cpads_csv()


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    description: str
    output_files: tuple[str, ...]


MODULE_SPECS = {
    "data-wrangling": ModuleSpec(
        name="data-wrangling",
        description="Canonicalize and validate the real CPADS PUMF input.",
        output_files=("data_na_summary.csv", "data_wrangling_log.csv"),
    ),
    "descriptive-statistics": ModuleSpec(
        name="descriptive-statistics",
        description="Survey-weighted prevalence and probability summaries.",
        output_files=("binomial_summaries.csv", "binomial_summaries_survey_weighted.csv", "probability_estimates.csv"),
    ),
    "distribution-tests": ModuleSpec(
        name="distribution-tests",
        description="Distributional diagnostics, correlations, and CLT checks.",
        output_files=("distribution_tests.csv", "alcohol_correlation_matrix.csv", "clt_convergence.csv"),
    ),
    "frequentist-inference": ModuleSpec(
        name="frequentist-inference",
        description="Frequentist prevalence, effect-size, and hypothesis-test outputs.",
        output_files=(
            "frequentist_heavy_drinking_prevalence_ci.csv",
            "frequentist_effect_sizes.csv",
            "frequentist_hypothesis_tests.csv",
        ),
    ),
    "bayesian-inference": ModuleSpec(
        name="bayesian-inference",
        description="Beta-binomial Bayesian summaries for key CPADS endpoints.",
        output_files=(
            "bayesian_posterior_summaries.csv",
            "bayesian_bayes_factors.csv",
            "bayesian_vs_frequentist_ci.csv",
        ),
    ),
    "power-design": ModuleSpec(
        name="power-design",
        description="Survey-weighted power planning summaries from real CPADS data.",
        output_files=(
            "power_summary.csv",
            "power_two_proportion_gender.csv",
            "power_one_proportion_grid.csv",
            "power_ebac_endpoint_anchors.csv",
            "power_gpower_reference_two_group.csv",
            "power_interaction_assumptions.csv",
            "power_interaction_feasibility_flags.csv",
            "power_interaction_group_allocations.csv",
            "power_interaction_imbalance_penalty.csv",
            "power_interaction_pairwise_details.csv",
            "power_interaction_sample_size_targets.csv",
            "randomization_block_blueprints.csv",
            "randomization_schedule_example_heavy_drinking_30d.csv",
            "randomization_schedule_example_ebac_legal.csv",
            "randomization_schedule_example_ebac_tot.csv",
        ),
    ),
    "logistic-models": ModuleSpec(
        name="logistic-models",
        description="Survey-weighted logistic models for heavy drinking.",
        output_files=(
            "logistic_odds_ratios.csv",
            "logistic_interaction_odds_ratios.csv",
            "logistic_interaction_tests.csv",
            "logistic_smote_status.csv",
            "logistic_smote_odds_ratios.csv",
        ),
    ),
    "model-comparison": ModuleSpec(
        name="model-comparison",
        description="Nested model comparison for heavy drinking models.",
        output_files=(
            "model_comparison_summary.csv",
            "model_comparison_full_coefs.csv",
            "model_comparison_interaction.csv",
            "model_comparison_wald_tests.csv",
        ),
    ),
    "regression-models": ModuleSpec(
        name="regression-models",
        description="Weighted regression models for eBAC outcomes.",
        output_files=("regression_coefficients.csv", "regression_model_comparison.csv"),
    ),
    "propensity-scores": ModuleSpec(
        name="propensity-scores",
        description="Propensity/IPW workflow for cannabis and heavy drinking.",
        output_files=("ipw_results.csv", "ipw_diagnostics.csv"),
    ),
    "causal-estimators": ModuleSpec(
        name="causal-estimators",
        description="Causal-estimator comparison across IPW, outcome-regression, and AIPW.",
        output_files=("causal_estimator_comparison.csv",),
    ),
    "treatment-effects": ModuleSpec(
        name="treatment-effects",
        description="ATE/ATT/ATC and subgroup treatment-effect summaries.",
        output_files=("treatment_effects_summary.csv", "cate_subgroup_estimates.csv"),
    ),
    "dag-specification": ModuleSpec(
        name="dag-specification",
        description="DAG and official-document alignment checklist outputs.",
        output_files=("official_doc_alignment_checklist.csv",),
    ),
    "meta-synthesis": ModuleSpec(
        name="meta-synthesis",
        description="Narrative synthesis outputs for study integration and interpretation.",
        output_files=("10_methods_results_paper.md", "11_interpretation.md"),
    ),
    "ebac-core": ModuleSpec(
        name="ebac-core",
        description="Core eBAC weighted, missingness, and model outputs.",
        output_files=(
            "ebac_data_quality_checks.csv",
            "ebac_distribution_unweighted.csv",
            "ebac_model_samples.csv",
            "ebac_weighted_summaries.csv",
            "ebac_missingness_weighted.csv",
            "ebac_missingness_or.csv",
            "ebac_missingness_or_eligible_drinkers.csv",
            "ebac_logistic_or_primary.csv",
            "ebac_linear_coefficients_primary.csv",
            "ebac_logistic_or_sensitivity_with_heavy.csv",
            "ebac_linear_coefficients_sensitivity_with_heavy.csv",
        ),
    ),
    "ebac-selection-adjustment-ipw": ModuleSpec(
        name="ebac-selection-adjustment-ipw",
        description="Selection-adjusted eBAC IPW workflow.",
        output_files=(
            "ebac_ipw_weight_diagnostics.csv",
            "ebac_ipw_logistic_or.csv",
            "ebac_ipw_linear_coefficients.csv",
            "ebac_ipw_cannabis_comparison.csv",
            "ebac_ipw_observation_model_or.csv",
            "ebac_ipw_covariate_balance.csv",
            "ebac_final_ipw_diagnostics.csv",
            "ebac_final_ipw_or.csv",
            "ebac_final_ipw_linear.csv",
            "ebac_final_ipw_comparison.csv",
        ),
    ),
    "ebac-integrations": ModuleSpec(
        name="ebac-integrations",
        description="Integrated eBAC final-summary outputs.",
        output_files=(
            "ebac_final_domain_samples.csv",
            "ebac_final_formula_input_audit.csv",
            "ebac_final_formula_validation.csv",
            "ebac_final_interaction_tests.csv",
            "ebac_final_weighted_descriptives.csv",
            "ebac_final_weighted_linear.csv",
            "ebac_final_weighted_or.csv",
            "ebac_final_smote_compare.csv",
            "ebac_final_smote_or.csv",
            "ebac_final_smote_status.csv",
            "ebac_final_causal_effects.csv",
            "ebac_final_cate.csv",
            "ebac_final_consistency_checks.csv",
            "ebac_final_crosswalk_previous.csv",
            "ebac_final_dml_results.csv",
            "ebac_final_dml_status.csv",
            "ebac_final_key_summary.csv",
            "ebac_final_user_guide_variable_map.csv",
            "ebac_final_variable_audit.csv",
        ),
    ),
    "ebac-gender-smote-sensitivity": ModuleSpec(
        name="ebac-gender-smote-sensitivity",
        description="eBAC interaction and SMOTE-sensitivity status outputs.",
        output_files=(
            "ebac_gender_interaction_svy_or.csv",
            "ebac_gender_interaction_tests.csv",
            "ebac_gender_marginal_probs.csv",
            "ebac_smote_status.csv",
            "ebac_smote_or.csv",
            "ebac_smote_compare.csv",
        ),
    ),
    "figures": ModuleSpec(
        name="figures",
        description="Figure exports for the documented analysis workflow.",
        output_files=(
            "figures/balance_plot.pdf",
            "figures/bayesian_prior_posterior.pdf",
            "figures/bayesian_prior_posterior.png",
            "figures/bayesian_vs_frequentist_ci.pdf",
            "figures/bayesian_vs_frequentist_ci.png",
            "figures/binge_by_demographics.pdf",
            "figures/binge_by_demographics.png",
            "figures/binge_by_mental_health.pdf",
            "figures/binge_by_mental_health.png",
            "figures/cate_forest_plot.pdf",
            "figures/cate_forest_plot.png",
            "figures/dag_heavy_drinking.pdf",
            "figures/qq_plots.pdf",
        ),
    ),
    "tables": ModuleSpec(
        name="tables",
        description="HTML table exports for the documented analysis workflow.",
        output_files=("table1.html",),
    ),
    "final-report": ModuleSpec(
        name="final-report",
        description="Final report and output-audit summaries.",
        output_files=(
            "ebac_final_output_coverage.csv",
            "ebac_final_output_shapes.csv",
            "ebac_final_script_run_status.csv",
            "ebac_final_audit_checks.csv",
            "ebac_final_user_guide_excerpt.txt",
        ),
    ),
    "otis-analysis": ModuleSpec(
        name="otis-analysis",
        description="Ontario restrictive confinement causal analysis (DML, PSM, GLMM, AIPW).",
        output_files=("otis_descriptives.csv", "otis_alert_combos.csv", "otis_dml_results.csv", "otis_trends.csv"),
    ),
    "mapq-psychometrics": ModuleSpec(
        name="mapq-psychometrics",
        description="MAPQ psychometric validation (CTT, CFA, omega) + DML (gender → KS).",
        output_files=("mapq_reliability.csv", "mapq_factor_loadings.csv", "mapq_dml_results.csv"),
    ),
}


def list_modules() -> list[dict[str, object]]:
    """Return the currently implemented CPADS module surface."""
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "output_files": list(spec.output_files),
        }
        for spec in MODULE_SPECS.values()
    ]


def load_cpads_analysis_data(cpads_csv: str | Path = DEFAULT_CPADS_CSV) -> pd.DataFrame:
    """Load and canonicalize the real CPADS CSV into MOIRAIS analysis columns."""
    registry = DatasetRegistry(data_dir=".")
    registry.register_local_cpads(Path(cpads_csv).expanduser().resolve(), name="cpads_local")
    raw = registry.load("cpads_local")
    return canonicalize_cpads_frame(raw)


def _project_root() -> Path:
    # moirais/modules.py → moirais → py-package → tools → config → libexec → project
    return Path(__file__).resolve().parents[5]


def _rscript_bin() -> str | None:
    return shutil.which("Rscript")


def _load_written_outputs(module_name: str, output_dir: Path) -> dict[str, object]:
    outputs: dict[str, object] = {}
    for filename in MODULE_SPECS[module_name].output_files:
        path = output_dir / filename
        if not path.exists():
            continue
        if path.suffix.lower() == ".csv":
            outputs[path.stem] = pd.read_csv(path)
        elif path.suffix.lower() == ".txt":
            outputs[path.stem] = path.read_text()
    return outputs


def _run_r_module(
    module_name: str,
    *,
    cpads_csv: str | Path = DEFAULT_CPADS_CSV,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    rscript = _rscript_bin()
    if rscript is None:
        raise RuntimeError("Rscript is not available on PATH.")
    _tmp_ctx = None
    if output_dir is None:
        _tmp_ctx = tempfile.TemporaryDirectory(prefix=f"moirais-{module_name}-")
        output_dir = Path(_tmp_ctx.name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        cmd = [
            rscript,
            str(_project_root() / "libexec" / "config" / "tests" / "rtests" / "run_modules.R"),
            f"--modules={module_name}",
            f"--cpads-csv={cpads_csv}",
            f"--output-dir={output_dir}",
        ]
        proc = subprocess.run(cmd, cwd=_project_root(), check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"R-backed module run failed for {module_name}.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        return _load_written_outputs(module_name, output_dir)
    finally:
        if _tmp_ctx is not None:
            _tmp_ctx.cleanup()


def _write_outputs(outputs: dict[str, pd.DataFrame], output_dir: str | Path | None) -> dict[str, pd.DataFrame]:
    if output_dir is None:
        return outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in outputs.items():
        if isinstance(table, pd.DataFrame):
            table.to_csv(output_dir / f"{name}.csv", index=False)
    return outputs


def run_power_design_module(
    cpads_csv: str | Path = DEFAULT_CPADS_CSV,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Build real-data power summaries from the CPADS CSV."""
    frame = load_cpads_analysis_data(cpads_csv)
    analysis = frame.dropna(subset=["heavy_drinking_30d", "gender", "weight"]).copy()
    gender_summary = (
        analysis.groupby("gender", dropna=True)
        .apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "weighted_prevalence": float((g["heavy_drinking_30d"] * g["weight"]).sum() / g["weight"].sum()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    pair_rows: list[dict[str, float]] = []
    power_grid_rows: list[dict[str, float]] = []
    power_tool = NormalIndPower()
    sample_sizes = [200, 400, 600, 800, 1000, 1500, 2000]

    if len(gender_summary) >= 2:
        ref = gender_summary.iloc[0]
        for _, other in gender_summary.iloc[1:].iterrows():
            effect = float(proportion_effectsize(ref["weighted_prevalence"], other["weighted_prevalence"]))
            for n_total in sample_sizes:
                n_per_group = n_total / 2
                power = float(power_tool.power(effect_size=abs(effect), nobs1=n_per_group, alpha=0.05, ratio=1.0))
                power_grid_rows.append(
                    {
                        "group1": ref["gender"],
                        "group2": other["gender"],
                        "n_total": n_total,
                        "n_per_group": n_per_group,
                        "effect_size_h": effect,
                        "power": power,
                    }
                )
            pair_rows.append(
                {
                    "group1": ref["gender"],
                    "group2": other["gender"],
                    "p1": ref["weighted_prevalence"],
                    "p2": other["weighted_prevalence"],
                    "effect_size_h": effect,
                    "n1": ref["n"],
                    "n2": other["n"],
                }
            )

    overall_prev = float((analysis["heavy_drinking_30d"] * analysis["weight"]).sum() / analysis["weight"].sum())

    power_summary = pd.DataFrame(
        [
            {"metric": "analysis_n", "value": float(len(analysis))},
            {"metric": "heavy_drinking_prevalence_weighted", "value": overall_prev},
            {"metric": "gender_levels_used", "value": float(gender_summary["gender"].nunique())},
        ]
    )

    # ── eBAC endpoint power anchors ──────────────────────────────────
    ebac_endpoints = ["ebac_legal", "ebac_tot"]
    anchor_rows = []
    for ep in ebac_endpoints:
        if ep in analysis.columns:
            ep_prev = (
                float(
                    (analysis[ep].dropna() * analysis.loc[analysis[ep].notna(), "weight"]).sum()
                    / analysis.loc[analysis[ep].notna(), "weight"].sum()
                )
                if analysis[ep].notna().any()
                else 0.0
            )
            for n in sample_sizes:
                try:
                    es = proportion_effectsize(ep_prev, ep_prev * 0.85)
                    pwr = float(power_tool.power(effect_size=abs(es), nobs1=n / 2, alpha=0.05, ratio=1.0))
                except Exception:
                    pwr = float("nan")
                anchor_rows.append({"endpoint": ep, "prevalence": ep_prev, "n_total": n, "power": pwr})

    # ── G*Power reference two-group ──────────────────────────────────
    gpower_rows = []
    for es_label, es_val in [("small", 0.2), ("medium", 0.5), ("large", 0.8)]:
        for n in sample_sizes:
            pwr = float(power_tool.power(effect_size=es_val, nobs1=n / 2, alpha=0.05, ratio=1.0))
            gpower_rows.append({"effect_size_label": es_label, "cohens_h": es_val, "n_total": n, "power": pwr})

    # ── Interaction power analysis ───────────────────────────────────
    interaction_assumptions = []
    interaction_feasibility = []
    interaction_allocations = []
    interaction_imbalance = []
    interaction_pairwise = []
    interaction_targets = []

    if "cannabis_any_use" in analysis.columns and len(gender_summary) >= 2:
        cannabis_prev = float(analysis["cannabis_any_use"].mean())
        for _, grow in gender_summary.iterrows():
            g = grow["gender"]
            g_mask = analysis["gender"] == g
            g_n = int(g_mask.sum())
            g_prev = grow["weighted_prevalence"]
            c_among_g = float(analysis.loc[g_mask, "cannabis_any_use"].mean()) if g_n > 0 else 0.0

            interaction_assumptions.append(
                {
                    "gender": g,
                    "n": g_n,
                    "heavy_drinking_prev": g_prev,
                    "cannabis_prev_in_group": c_among_g,
                    "overall_cannabis_prev": cannabis_prev,
                }
            )

            # Feasibility: need ≥30 per cell for 2×2 interaction
            n_cells = max(1, int(g_n * c_among_g)), max(1, int(g_n * (1 - c_among_g)))
            min_cell = min(n_cells)
            interaction_feasibility.append(
                {
                    "gender": g,
                    "min_cell_n": min_cell,
                    "feasible": "Yes" if min_cell >= 30 else "No",
                    "flag": "" if min_cell >= 30 else "cell_too_small",
                }
            )

            interaction_allocations.append(
                {
                    "gender": g,
                    "n_total": g_n,
                    "n_cannabis": n_cells[0],
                    "n_no_cannabis": n_cells[1],
                    "ratio": round(n_cells[0] / n_cells[1], 3) if n_cells[1] > 0 else float("nan"),
                }
            )

            # Imbalance penalty
            ratio = n_cells[0] / n_cells[1] if n_cells[1] > 0 else 1.0
            penalty = abs(1.0 - ratio) * 0.1
            interaction_imbalance.append(
                {"gender": g, "allocation_ratio": round(ratio, 3), "penalty_factor": round(penalty, 4)}
            )

        # Pairwise interaction power
        genders = gender_summary["gender"].tolist()
        for i in range(len(genders)):
            for j in range(i + 1, len(genders)):
                g1, g2 = genders[i], genders[j]
                p1 = float(gender_summary.loc[gender_summary["gender"] == g1, "weighted_prevalence"].iloc[0])
                p2 = float(gender_summary.loc[gender_summary["gender"] == g2, "weighted_prevalence"].iloc[0])
                diff = abs(p1 - p2)
                es = proportion_effectsize(p1, p2) if p1 > 0 and p2 > 0 else 0.0
                for n in sample_sizes:
                    try:
                        pwr = float(power_tool.power(effect_size=abs(es) * 0.5, nobs1=n / 4, alpha=0.05, ratio=1.0))
                    except Exception:
                        pwr = float("nan")
                    interaction_pairwise.append(
                        {
                            "gender1": g1,
                            "gender2": g2,
                            "prevalence_diff": diff,
                            "interaction_effect_h": round(es * 0.5, 4),
                            "n_total": n,
                            "power": pwr,
                        }
                    )

        # Target sample sizes for 80% power
        for i in range(len(genders)):
            for j in range(i + 1, len(genders)):
                g1, g2 = genders[i], genders[j]
                p1 = float(gender_summary.loc[gender_summary["gender"] == g1, "weighted_prevalence"].iloc[0])
                p2 = float(gender_summary.loc[gender_summary["gender"] == g2, "weighted_prevalence"].iloc[0])
                es = proportion_effectsize(p1, p2) * 0.5 if p1 > 0 and p2 > 0 else 0.0
                try:
                    n80 = (
                        float(power_tool.solve_power(effect_size=abs(es), alpha=0.05, power=0.80, ratio=1.0))
                        if es != 0
                        else float("nan")
                    )
                except Exception:
                    n80 = float("nan")
                interaction_targets.append(
                    {
                        "gender1": g1,
                        "gender2": g2,
                        "target_power": 0.80,
                        "interaction_effect_h": round(es, 4),
                        "required_n_per_group": round(n80, 0) if not np.isnan(n80) else float("nan"),
                        "required_n_total": round(n80 * 4, 0) if not np.isnan(n80) else float("nan"),
                    }
                )

    # ── Randomization schedules ──────────────────────────────────────
    rng = np.random.RandomState(42)
    block_sizes = [4, 6, 8]
    block_rows = []
    for bs in block_sizes:
        n_treated = bs // 2
        n_control = bs - n_treated
        block_rows.append(
            {
                "block_size": bs,
                "n_treated_per_block": n_treated,
                "n_control_per_block": n_control,
                "n_blocks_for_200": 200 // bs,
                "n_blocks_for_1000": 1000 // bs,
            }
        )

    def _make_schedule(name: str, n: int = 200) -> pd.DataFrame:
        schedule = []
        seq_id = 1
        block_id = 1
        while seq_id <= n:
            bs = rng.choice(block_sizes)
            assignments = ["treatment"] * (bs // 2) + ["control"] * (bs - bs // 2)
            rng.shuffle(assignments)
            for a in assignments:
                if seq_id > n:
                    break
                schedule.append({"sequence_id": seq_id, "block_id": block_id, "block_size": bs, "assignment": a})
                seq_id += 1
            block_id += 1
        return pd.DataFrame(schedule)

    outputs = {
        "power_summary": power_summary,
        "power_two_proportion_gender": pd.DataFrame(pair_rows),
        "power_one_proportion_grid": pd.DataFrame(power_grid_rows),
        "power_ebac_endpoint_anchors": pd.DataFrame(anchor_rows),
        "power_gpower_reference_two_group": pd.DataFrame(gpower_rows),
        "power_interaction_assumptions": pd.DataFrame(interaction_assumptions),
        "power_interaction_feasibility_flags": pd.DataFrame(interaction_feasibility),
        "power_interaction_group_allocations": pd.DataFrame(interaction_allocations),
        "power_interaction_imbalance_penalty": pd.DataFrame(interaction_imbalance),
        "power_interaction_pairwise_details": pd.DataFrame(interaction_pairwise),
        "power_interaction_sample_size_targets": pd.DataFrame(interaction_targets),
        "randomization_block_blueprints": pd.DataFrame(block_rows),
        "randomization_schedule_example_heavy_drinking_30d": _make_schedule("heavy_drinking_30d"),
        "randomization_schedule_example_ebac_legal": _make_schedule("ebac_legal"),
        "randomization_schedule_example_ebac_tot": _make_schedule("ebac_tot"),
    }
    return _write_outputs(outputs, output_dir)


def _run_ebac_gender_smote_sensitivity(
    data: pd.DataFrame,
    *,
    treatment: str = "cannabis_any_use",
    binary_outcome: str = "ebac_legal",
    weight_col: str = "weight",
) -> dict[str, pd.DataFrame]:
    """eBAC gender interaction and SMOTE sensitivity analysis."""
    import statsmodels.api as sm

    from .ml import apply_smote
    from .survey import SurveyDesign

    covariates = ["age_group", "gender", "province_region", "mental_health", "physical_health"]
    required = [binary_outcome, treatment, weight_col, *covariates]
    frame = data.loc[:, [c for c in required if c in data.columns]].dropna().copy()

    design = SurveyDesign(frame, weights_col=weight_col)

    # Gender interaction survey-weighted logistic OR
    formula = f"{binary_outcome} ~ {treatment} + " + " + ".join(covariates) + f" + {treatment}:gender"
    try:
        fit = design.svyglm(formula, family=sm.families.Binomial())
        conf = fit.conf_int()
        or_rows = []
        for term in fit.params.index:
            or_rows.append(
                {
                    "term": term,
                    "log_odds": float(fit.params[term]),
                    "SE": float(fit.bse[term]),
                    "OR": float(np.exp(np.clip(fit.params[term], -700, 700))),
                    "OR_lower95": float(np.exp(np.clip(conf.loc[term, 0], -700, 700))),
                    "OR_upper95": float(np.exp(np.clip(conf.loc[term, 1], -700, 700))),
                    "p_value": float(fit.pvalues[term]),
                }
            )
        gender_or = pd.DataFrame(or_rows)

        # Interaction Wald test
        int_terms = [t for t in fit.params.index if ":" in t and treatment in t and "gender" in t]
        if int_terms:
            contrast = np.zeros((len(int_terms), len(fit.params)))
            idx_list = list(fit.params.index)
            for ri, t in enumerate(int_terms):
                contrast[ri, idx_list.index(t)] = 1.0
            wald = fit.wald_test(contrast, scalar=True)
            from .investigation import _scalarize

            stat = _scalarize(wald.statistic)
            pval = _scalarize(wald.pvalue)
        else:
            stat, pval = 0.0, 1.0

        interaction_tests = pd.DataFrame(
            [
                {
                    "test": f"{treatment}:gender (joint Wald)",
                    "F_stat": stat,
                    "df_num": len(int_terms),
                    "p_value": pval,
                }
            ]
        )
    except Exception:
        gender_or = pd.DataFrame(columns=["term", "log_odds", "SE", "OR", "OR_lower95", "OR_upper95", "p_value"])
        interaction_tests = pd.DataFrame(columns=["test", "F_stat", "df_num", "p_value"])

    # Marginal predicted probabilities by gender
    marginal_rows = []
    for g_level, g_sub in frame.groupby("gender"):
        prev = (
            float((g_sub[binary_outcome] * g_sub[weight_col]).sum() / g_sub[weight_col].sum())
            if g_sub[weight_col].sum() > 0
            else 0.0
        )
        marginal_rows.append({"gender": g_level, "n": len(g_sub), "marginal_prob": prev})
    marginal_probs = pd.DataFrame(marginal_rows)

    # SMOTE sensitivity
    y = frame[binary_outcome].astype(int)
    X = pd.get_dummies(frame[[treatment, *covariates]], drop_first=True, dtype=float)
    X_res, y_res, smote_info = apply_smote(X, y)
    smote_status = pd.DataFrame([smote_info])

    try:
        smote_fit = sm.GLM(y_res, sm.add_constant(X_res), family=sm.families.Binomial()).fit()
        smote_conf = smote_fit.conf_int()
        smote_or = pd.DataFrame(
            {
                "term": smote_fit.params.index,
                "log_odds": smote_fit.params.values,
                "SE": smote_fit.bse.values,
                "OR": np.exp(np.clip(smote_fit.params.values, -700, 700)),
                "OR_lower95": np.exp(np.clip(smote_conf[0].values, -700, 700)),
                "OR_upper95": np.exp(np.clip(smote_conf[1].values, -700, 700)),
                "p_value": smote_fit.pvalues.values,
            }
        )
    except Exception:
        smote_or = pd.DataFrame(columns=["term", "log_odds", "SE", "OR", "OR_lower95", "OR_upper95", "p_value"])

    # SMOTE comparison: original vs SMOTE odds ratios
    compare_rows = []
    for term in gender_or["term"].values:
        orig = gender_or.loc[gender_or["term"] == term]
        resampled = smote_or.loc[smote_or["term"] == term] if term in smote_or["term"].values else pd.DataFrame()
        compare_rows.append(
            {
                "term": term,
                "OR_original": float(orig["OR"].iloc[0]) if len(orig) else float("nan"),
                "OR_smote": float(resampled["OR"].iloc[0]) if len(resampled) else float("nan"),
                "p_original": float(orig["p_value"].iloc[0]) if len(orig) else float("nan"),
                "p_smote": float(resampled["p_value"].iloc[0]) if len(resampled) else float("nan"),
            }
        )
    smote_compare = pd.DataFrame(compare_rows)

    return {
        "ebac_gender_interaction_svy_or": gender_or,
        "ebac_gender_interaction_tests": interaction_tests,
        "ebac_gender_marginal_probs": marginal_probs,
        "ebac_smote_status": smote_status,
        "ebac_smote_or": smote_or,
        "ebac_smote_compare": smote_compare,
    }


def _load_dataset_frame(
    dataset_key: str | None = None,
    cpads_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Load a dataset by key, path, or from the DB.

    Supports three modes:
    1. **Catalog key** — e.g. "ocp21", "hibp" → loads from SQLite DB
    2. **Arbitrary CSV/XLSX path** — e.g. "/tmp/my_data.csv" → reads directly
    3. **Default** — CPADS CSV (backward compat when no key/path given)

    Parameters
    ----------
    dataset_key : str, optional
        Short key from DATASET_CATALOG, OR a file path to any CSV/XLSX.
        When it's a file path, the data is loaded directly without
        column validation (dataset-agnostic mode).
    cpads_csv : str or Path, optional
        Legacy: direct path to a CPADS CSV file.
    """
    if dataset_key:
        # Check if it's a file path first
        key_path = Path(dataset_key)
        if key_path.suffix in {".csv", ".xlsx", ".xls", ".tsv", ".parquet"} or key_path.exists():
            if key_path.exists():
                if key_path.suffix == ".csv":
                    return pd.read_csv(key_path)
                elif key_path.suffix == ".tsv":
                    return pd.read_csv(key_path, sep="\t")
                elif key_path.suffix in {".xlsx", ".xls"}:
                    return pd.read_excel(key_path)
                elif key_path.suffix == ".parquet":
                    return pd.read_parquet(key_path)
            else:
                raise FileNotFoundError(f"Dataset file not found: {dataset_key}")

        # Check if it's a catalog key
        from .data import DATASET_CATALOG, load_dataset

        if dataset_key in DATASET_CATALOG:
            return load_dataset(dataset_key)

        # Not a known key and not a file path — try fuzzy match
        matches = [k for k in DATASET_CATALOG if dataset_key.lower() in k]
        if matches:
            return load_dataset(matches[0])

        raise ValueError(
            f"Unknown dataset: {dataset_key}. "
            f"Pass a catalog key ({', '.join(sorted(list(DATASET_CATALOG)[:5]))}...) "
            f"or a file path (CSV/XLSX/TSV/Parquet)."
        )

    # Default: CPADS via CSV (backward compat)
    csv_path = cpads_csv or DEFAULT_CPADS_CSV
    return load_cpads_analysis_data(csv_path)


def run_module(
    module_name: str,
    *,
    cpads_csv: str | Path = DEFAULT_CPADS_CSV,
    dataset_key: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Run one implemented MOIRAIS module.

    Parameters
    ----------
    module_name : str
        Name from MODULE_SPECS (e.g. "power-design").
    cpads_csv : str or Path
        Legacy: direct path to CPADS CSV.
    dataset_key : str, optional
        Short key from DATASET_CATALOG. When provided, loads from DB
        instead of the CSV path. Default: None (uses cpads_csv).
    output_dir : str or Path, optional
        Directory for CSV outputs.
    """
    if module_name not in MODULE_SPECS:
        valid = ", ".join(sorted(MODULE_SPECS))
        raise ValueError(f"Unknown module: {module_name}. Valid modules: {valid}")

    try:
        return _run_r_module(module_name, cpads_csv=cpads_csv, output_dir=output_dir)
    except Exception:
        if module_name not in {
            "power-design",
            "logistic-models",
            "model-comparison",
            "propensity-scores",
            "treatment-effects",
            "ebac-selection-adjustment-ipw",
            "ebac-gender-smote-sensitivity",
        }:
            raise

    if module_name == "power-design":
        return run_power_design_module(cpads_csv, output_dir=output_dir)

    frame = _load_dataset_frame(dataset_key=dataset_key, cpads_csv=cpads_csv)
    if module_name == "logistic-models":
        outputs = run_weighted_logistic_analysis(frame)
    elif module_name == "model-comparison":
        outputs = compare_nested_logistic_models(frame)
    elif module_name == "propensity-scores":
        outputs = run_propensity_ipw_analysis(frame)
    elif module_name == "treatment-effects":
        outputs = run_treatment_effects_analysis(frame)
    elif module_name == "ebac-selection-adjustment-ipw":
        outputs = run_ebac_selection_ipw_analysis(frame)
    elif module_name == "ebac-gender-smote-sensitivity":
        outputs = _run_ebac_gender_smote_sensitivity(frame)
    else:  # pragma: no cover
        raise AssertionError(f"Unhandled module: {module_name}")

    csv_outputs = {
        name: table for name, table in outputs.items() if isinstance(table, pd.DataFrame) and name != "analysis_frame"
    }
    return _write_outputs(csv_outputs, output_dir)


def run_modules(
    module_names: list[str] | None = None,
    *,
    cpads_csv: str | Path = DEFAULT_CPADS_CSV,
    dataset_key: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Run multiple implemented modules and return their output tables."""
    module_names = module_names or list(MODULE_SPECS)
    return {
        name: run_module(
            name,
            cpads_csv=cpads_csv,
            dataset_key=dataset_key,
            output_dir=output_dir,
        )
        for name in module_names
    }
