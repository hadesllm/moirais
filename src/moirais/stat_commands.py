"""
Central command registry for MOIRAIS StatScreen and REPL.

Wraps all 620+ backend functions as user-facing commands plus aliases,
wrangling helpers, R bridge commands, and compound workflows.
Both StatScreen and REPL dispatch through this registry.

All backend imports are lazy (inside handler bodies) to avoid loading
scipy/numpy/pandas at import time.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Registry infrastructure
# ---------------------------------------------------------------------------


@dataclass
class StatCommand:
    """A single user-facing command."""

    name: str
    category: str
    usage: str
    description: str
    handler_stat: Callable  # (parts, log, store) -> None
    handler_repl: Callable  # plain function for REPL namespace
    aliases: list[str] = field(default_factory=list)
    module: str = ""
    is_compound: bool = False
    is_r_bridge: bool = False


COMMAND_REGISTRY: dict[str, StatCommand] = {}
ALIAS_MAP: dict[str, str] = {}
CATEGORIES: dict[str, list[str]] = {}


def register(cmd: StatCommand) -> None:
    """Register a command and its aliases."""
    COMMAND_REGISTRY[cmd.name] = cmd
    CATEGORIES.setdefault(cmd.category, []).append(cmd.name)
    for alias in cmd.aliases:
        ALIAS_MAP[alias] = cmd.name


def resolve(name: str) -> StatCommand | None:
    """Look up a command by name or alias."""
    if name in COMMAND_REGISTRY:
        return COMMAND_REGISTRY[name]
    canonical = ALIAS_MAP.get(name)
    if canonical:
        return COMMAND_REGISTRY.get(canonical)
    return None


def all_command_names() -> list[str]:
    """All names including aliases, sorted."""
    return sorted(set(list(COMMAND_REGISTRY.keys()) + list(ALIAS_MAP.keys())))


def commands_by_category() -> dict[str, list[StatCommand]]:
    """Commands grouped by category."""
    result: dict[str, list[StatCommand]] = {}
    for cat, names in sorted(CATEGORIES.items()):
        result[cat] = [COMMAND_REGISTRY[n] for n in names if n in COMMAND_REGISTRY]
    return result


# ---------------------------------------------------------------------------
# Generic handler factories
# ---------------------------------------------------------------------------


def _make_stat_handler(mod_name: str, func_name: str) -> Callable:
    """Create a StatScreen handler that calls a backend function generically.

    The handler parses command parts and calls the function, displaying
    the result in the RichLog.
    """

    def handler(parts: list[str], log: Any, store: Callable) -> None:
        import importlib

        mod = importlib.import_module(f"moirais.{mod_name}")
        fn = getattr(mod, func_name)
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())

        # Collect args from parts[1:] — skip the command name
        args = parts[1:]

        # Check if first param expects a DataFrame/array
        needs_df = False
        if params and params[0].name in (
            "data",
            "df",
            "x",
            "sample",
            "time",
            "outcome",
            "y",
        ):
            needs_df = True

        if needs_df and len(args) >= 1:
            import pandas as pd

            csv_path = args[0]
            try:
                if csv_path.endswith((".xlsx", ".xls")):
                    # Check for sheet=Name in remaining args
                    sheet = 0
                    remaining_pre = list(args[1:])
                    for i, a in enumerate(remaining_pre):
                        if a.startswith("sheet="):
                            sheet = a.split("=", 1)[1]
                            remaining_pre.pop(i)
                            break
                    df = pd.read_excel(csv_path, sheet_name=sheet)
                    args = [args[0]] + remaining_pre
                else:
                    df = pd.read_csv(csv_path)
            except Exception as e:
                log.write(f"[red]Cannot read: {csv_path} ({e})[/red]")
                return
            # Pass remaining args
            remaining = args[1:]
            # If function expects column names, extract from df
            try:
                result = fn(df, *remaining)
            except TypeError:
                # Try passing just the df
                try:
                    result = fn(df)
                except Exception as e:
                    log.write(f"[red]Error: {e}[/red]")
                    return
        elif args:
            # Try to convert numeric args
            converted = []
            for a in args:
                try:
                    converted.append(float(a))
                except ValueError:
                    converted.append(a)
            try:
                result = fn(*converted)
            except Exception as e:
                log.write(f"[red]Error: {e}[/red]")
                return
        else:
            try:
                result = fn()
            except Exception:
                log.write(f"[yellow]Usage: {func_name} <args>[/yellow]")
                log.write(f"[dim]{fn.__doc__.splitlines()[0] if fn.__doc__ else ''}[/dim]")
                return

        # Display result
        log.write(f"\n[bold cyan]{func_name}[/bold cyan]")
        _display_result(result, log)
        store(str(result))

    return handler


def _make_repl_handler(mod_name: str, func_name: str) -> Callable:
    """Create a REPL handler that directly calls the backend function."""

    def handler(*args: Any, **kwargs: Any) -> Any:
        import importlib
        import inspect as _inspect

        mod = importlib.import_module(f"moirais.{mod_name}")
        fn = getattr(mod, func_name)
        try:
            return fn(*args, **kwargs)
        except TypeError as e:
            if "required" in str(e) or "positional" in str(e):
                sig = _inspect.signature(fn)
                params = ", ".join(
                    p.name
                    for p in sig.parameters.values()
                    if p.default is _inspect.Parameter.empty
                    and p.kind not in (_inspect.Parameter.VAR_POSITIONAL, _inspect.Parameter.VAR_KEYWORD)
                )
                print(f"Usage: {func_name}({params})")
                doc = fn.__doc__
                if doc:
                    print(f"  {doc.strip().splitlines()[0]}")
                return None
            raise

    # Copy docstring
    try:
        import importlib

        _mod = importlib.import_module(f"moirais.{mod_name}")
        _fn = getattr(_mod, func_name)
        handler.__doc__ = _fn.__doc__
        handler.__name__ = func_name
    except Exception:
        handler.__doc__ = f"{func_name} from moirais.{mod_name}"
        handler.__name__ = func_name

    return handler


def _display_result(result: Any, log: Any) -> None:
    """Display a result object in the RichLog."""
    import pandas as pd

    if isinstance(result, pd.DataFrame):
        log.write(result.to_string(max_rows=30))
    elif hasattr(result, "__dataclass_fields__"):
        for f in result.__dataclass_fields__:
            val = getattr(result, f)
            if isinstance(val, float):
                log.write(f"  {f}: {val:.6f}")
            else:
                log.write(f"  {f}: {val}")
    elif hasattr(result, "__dict__") and not isinstance(result, type):
        for k, v in result.__dict__.items():
            if not k.startswith("_"):
                if isinstance(v, float):
                    log.write(f"  {k}: {v:.6f}")
                else:
                    log.write(f"  {k}: {v}")
    elif isinstance(result, dict):
        for k, v in result.items():
            if isinstance(v, float):
                log.write(f"  {k}: {v:.6f}")
            else:
                log.write(f"  {k}: {v}")
    elif isinstance(result, (list, tuple)):
        for item in result[:20]:
            log.write(f"  {item}")
        if len(result) > 20:
            log.write(f"  ... ({len(result)} total)")
    else:
        log.write(f"  {result}")


# ---------------------------------------------------------------------------
# Auto-registration: wrap all public functions from a module
# ---------------------------------------------------------------------------


def _auto_register(
    mod_name: str,
    category: str,
    aliases: dict[str, list[str]] | None = None,
    exclude: set[str] | None = None,
    descriptions: dict[str, str] | None = None,
) -> int:
    """Auto-register all public functions/classes from moirais.<mod_name>.

    Returns the number of commands registered.
    """
    import importlib

    aliases = aliases or {}
    exclude = exclude or set()
    descriptions = descriptions or {}
    count = 0

    try:
        mod = importlib.import_module(f"moirais.{mod_name}")
    except ImportError:
        return 0

    for name in sorted(dir(mod)):
        if name.startswith("_") or name in exclude:
            continue
        obj = getattr(mod, name)

        # Only register functions and classes defined in this module
        obj_module = getattr(obj, "__module__", "")
        if obj_module != f"moirais.{mod_name}":
            continue

        if not (callable(obj) or isinstance(obj, type)):
            continue

        # Get description from docstring or override
        desc = descriptions.get(name, "")
        if not desc:
            doc = getattr(obj, "__doc__", "") or ""
            first_line = doc.strip().split("\n")[0] if doc.strip() else ""
            desc = first_line[:100] if first_line else f"{name} from {mod_name}"

        # Build usage string from signature
        if inspect.isfunction(obj):
            try:
                sig = inspect.signature(obj)
                params = [
                    p.name
                    for p in sig.parameters.values()
                    if p.name != "self"
                    and p.default is inspect.Parameter.empty
                    and p.kind
                    not in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    )
                ]
                usage = f"{name} {' '.join(f'<{p}>' for p in params[:4])}"
            except (ValueError, TypeError):
                usage = f"{name} ..."
        else:
            usage = f"{name}()"

        register(
            StatCommand(
                name=name,
                category=category,
                usage=usage,
                description=desc,
                handler_stat=_make_stat_handler(mod_name, name),
                handler_repl=_make_repl_handler(mod_name, name),
                aliases=aliases.get(name, []),
                module=mod_name,
            )
        )
        count += 1

    return count


# ---------------------------------------------------------------------------
# Module registrations — called at module load time
# ---------------------------------------------------------------------------


def _register_all_backend_modules() -> int:
    """Register all 26 backend modules. Returns total command count."""
    total = 0

    # 1. statistics.py (41 fns)
    total += _auto_register(
        "statistics",
        "Statistics",
        aliases={
            "one_sample_ttest": ["ttest1"],
            "two_sample_ttest": ["ttest_ind"],
            "welch_ttest": ["welch"],
            "paired_ttest": ["paired_t"],
            "one_way_anova": ["anova1"],
            "two_way_anova": ["anova2"],
            "kruskal_wallis": ["kw_test"],
            "chi2_independence": ["chi2_ind"],
            "chi2_goodness_of_fit": ["chi2_gof"],
            "mann_whitney_u": ["mwu"],
            "wilcoxon_signed_rank": ["wsr"],
            "ks_test_one_sample": ["ks1"],
            "ks_test_two_sample": ["ks2"],
            "shapiro_wilk": ["sw_test"],
            "normality_battery": ["norm_tests"],
            "variance_equality_battery": ["var_tests"],
            "auto_test": ["autotest"],
            "pearson_correlation": ["pearson"],
            "spearman_correlation": ["spearman"],
            "kendall_correlation": ["kendall"],
            "correlation_matrix": ["corrmatrix"],
        },
    )

    # 2. inference.py (57 fns)
    total += _auto_register(
        "inference",
        "Inference",
        aliases={
            "one_sample_t_test": ["inf_ttest1"],
            "two_sample_t_test": ["inf_ttest2"],
            "paired_t_test": ["inf_paired"],
            "chi_square_test": ["inf_chi2"],
            "bootstrap_ci": ["boot_ci"],
            "power_t_test": ["power_t"],
            "power_prop_test": ["power_prop"],
            "power_anova": ["power_f"],
            "sample_size_logistic": ["ss_logistic"],
        },
    )

    # 3. effect_sizes.py (39 fns + 1 class)
    total += _auto_register(
        "effect_sizes",
        "Effect Sizes",
        aliases={
            "cohens_d": ["cd"],
            "hedges_g": ["hg"],
            "glass_delta": ["gd"],
            "odds_ratio": ["or_calc"],
            "risk_ratio": ["rr_calc"],
            "risk_difference": ["rd_calc"],
            "number_needed_to_treat": ["nnt_calc"],
            "number_needed_to_harm": ["nnh_calc"],
            "cramers_v": ["cv_effect"],
            "fixed_effects_meta": ["fe_meta"],
            "random_effects_meta": ["re_meta"],
            "d_to_r": ["d2r"],
            "r_to_d": ["r2d"],
            "or_to_d": ["or2d"],
            "d_to_or": ["d2or"],
            "d_to_nnt": ["d2nnt"],
        },
    )

    # 4. survival.py (32 fns + 5 classes)
    total += _auto_register(
        "survival",
        "Survival",
        aliases={
            "kaplan_meier": ["km"],
            "nelson_aalen": ["na_est"],
            "logrank_test": ["logrank"],
            "cox_ph": ["cox"],
            "restricted_mean_survival_time": ["rmst"],
            "concordance_index": ["c_index"],
            "hazard_ratio": ["hr"],
            "aft_weibull": ["aft_w"],
            "aft_lognormal": ["aft_ln"],
            "aft_loglogistic": ["aft_ll"],
            "cumulative_incidence_function": ["cif"],
        },
    )

    # 5. matching.py (27 fns + 3 classes)
    total += _auto_register(
        "matching",
        "Matching",
        aliases={
            "match_nearest_neighbor": ["nn_match", "ps_nn"],
            "match_exact": ["exact_match"],
            "match_cem": ["cem"],
            "match_mahalanobis": ["maha_match"],
            "match_optimal_pair": ["opt_match"],
            "match_full": ["full_match"],
            "match_genetic": ["gen_match"],
            "match_variable_ratio": ["vr_match"],
            "match_cardinality": ["card_match"],
            "estimate_propensity_score": ["est_ps"],
            "balance_diagnostics": ["bal_diag"],
            "love_plot_data": ["love_data"],
            "estimate_att_matched": ["att_match"],
            "estimate_ate_matched": ["ate_match"],
            "estimate_atc_matched": ["atc_match"],
            "doubly_robust_matching": ["dr_match"],
            "subclassify": ["ps_subclass"],
            "entropy_balance": ["ebal"],
        },
    )

    # 6. causal.py (13 fns)
    total += _auto_register(
        "causal",
        "Causal",
        aliases={
            "compute_propensity_scores": ["ps_scores"],
            "calculate_ipw_weights": ["ipw_wts"],
            "estimate_aipw": ["aipw"],
            "estimate_att": ["att"],
            "estimate_atc": ["atc"],
            "estimate_cate": ["cate"],
            "estimate_gate": ["gate"],
            "estimate_late": ["late"],
            "estimate_irm": ["irm"],
            "estimate_double_ml": ["dml"],
        },
    )

    # 7. effects.py (6 fns)
    total += _auto_register(
        "effects",
        "Effects",
        aliases={
            "estimate_ate": ["ate_ols"],
            "estimate_plr": ["plr"],
            "estimate_pliv": ["pliv"],
            "estimate_ate_gcomputation": ["gcomp"],
            "sensitivity_rosenbaum": ["rosen_sens"],
            "e_value": ["eval_sens"],
        },
    )

    # 8. did.py (23 fns + 3 classes)
    total += _auto_register(
        "did",
        "DiD",
        aliases={
            "did_2x2": ["did2x2"],
            "event_study": ["es_did"],
            "staggered_did": ["stag_did"],
            "did_doubly_robust": ["dr_did"],
            "bacon_decomposition": ["bacon"],
            "synthetic_did": ["sdid"],
            "did_triple_difference": ["ddd"],
            "wild_cluster_bootstrap": ["wcb"],
        },
    )

    # 9. rdd.py (24 fns + 3 classes)
    total += _auto_register(
        "rdd",
        "RDD",
        aliases={
            "sharp_rdd": ["srdd"],
            "fuzzy_rdd": ["frdd"],
            "rdd_bias_corrected": ["bc_rdd"],
            "mccrary_test": ["mccrary"],
            "bandwidth_cct": ["bw_cct"],
            "bandwidth_ik": ["bw_ik"],
            "donut_rdd": ["drdd"],
            "kink_rdd": ["krdd"],
        },
    )

    # 10. iv.py (23 fns + 2 classes)
    total += _auto_register(
        "iv",
        "IV",
        aliases={
            "tsls": ["2sls"],
            "liml": ["iv_liml"],
            "gmm_iv": ["iv_gmm"],
            "wald_estimator": ["wald_iv"],
            "hausman_test": ["hausman"],
            "sargan_test": ["sargan"],
            "hansen_j_test": ["hansen_j"],
            "anderson_rubin_test": ["ar_test"],
            "durbin_wu_hausman": ["dwh"],
        },
    )

    # 11. weights.py (26 fns + 4 classes)
    total += _auto_register(
        "weights",
        "Weights",
        aliases={
            "poststratify": ["ps_weight"],
            "rake": ["ipf"],
            "trim_weights": ["wtrim"],
            "calibrate_to_totals": ["cal_wt"],
            "normalize_weights": ["wnorm"],
        },
    )

    # 12. missing.py (26 fns + 5 classes)
    total += _auto_register(
        "missing",
        "Missing Data",
        aliases={
            "mice": ["mi"],
            "littles_mcar_test": ["mcar_test"],
            "impute_mean": ["imp_mean"],
            "impute_median": ["imp_median"],
            "impute_mode": ["imp_mode"],
            "impute_locf": ["locf"],
            "impute_nocb": ["nocb"],
            "impute_regression": ["imp_reg"],
            "impute_hot_deck": ["hot_deck"],
            "rubins_rules": ["rubins"],
            "pool_mice_estimates": ["pool_mi"],
        },
    )

    # 13. multiple_testing.py (26 fns + 3 classes)
    total += _auto_register(
        "multiple_testing",
        "Multiple Testing",
        aliases={
            "benjamini_hochberg": ["bh_fdr"],
            "benjamini_yekutieli": ["by_fdr"],
            "storey_q": ["qvalue"],
            "fisher_combined": ["fisher_comb"],
            "stouffer_combined": ["stouffer"],
            "harmonic_mean_p": ["hmp"],
        },
    )

    # 14. reporting.py (31 fns + 6 classes)
    total += _auto_register(
        "reporting",
        "Reporting",
        aliases={
            "format_p_value": ["fmt_p"],
            "format_ci": ["fmt_ci"],
            "check_strobe_compliance": ["strobe_check"],
            "generate_full_report": ["full_report"],
            "compile_report": ["report"],
        },
    )

    # 15. viz.py (30 fns)
    total += _auto_register(
        "viz",
        "Visualization",
        aliases={
            "forest_plot": ["fp"],
            "funnel_plot": ["fnp"],
            "kaplan_meier_plot": ["km_plot"],
            "dag_plot": ["dag"],
            "qq_plot": ["qqp"],
            "correlation_heatmap": ["heatmap"],
            "missing_pattern_plot": ["miss_plot"],
            "propensity_score_plot": ["ps_plot"],
            "event_study_plot": ["es_plot"],
        },
    )

    # 16. export.py (17 fns + 2 classes)
    total += _auto_register(
        "export",
        "Export",
        aliases={
            "df_to_latex": ["to_latex"],
            "df_to_html": ["to_html"],
            "df_to_markdown": ["to_md"],
            "df_to_docx": ["to_docx"],
            "df_to_excel": ["to_xlsx"],
            "df_to_json": ["to_json"],
            "df_to_csv_with_meta": ["to_csv_meta"],
            "export_results_bundle": ["bundle"],
            "batch_export": ["bexport"],
        },
    )

    # 17. diagnostics.py (13 fns + 6 classes)
    total += _auto_register(
        "diagnostics",
        "Diagnostics",
        aliases={
            "compute_vif": ["vif_calc"],
            "compute_residuals": ["resid"],
            "compute_influence": ["influence"],
            "hosmer_lemeshow_test": ["hl_test"],
            "ramsey_reset_test": ["reset_test"],
            "full_diagnostics": ["full_diag"],
        },
    )

    # 18. bootstrap_methods.py (13 fns + 4 classes)
    total += _auto_register(
        "bootstrap_methods",
        "Bootstrap",
        aliases={
            "bootstrap": ["boot"],
            "parametric_bootstrap": ["pboot"],
            "wild_bootstrap": ["wboot"],
            "block_bootstrap": ["bboot"],
            "jackknife": ["jk"],
            "permutation_test": ["perm"],
            "paired_permutation_test": ["pperm"],
            "cross_validate": ["cv"],
            "leave_one_out_cv": ["loocv"],
        },
    )

    # 19. validation.py (13 fns + 11 classes)
    total += _auto_register(
        "validation",
        "Validation",
        aliases={
            "cross_validate": ["val_cv"],
            "decision_curve_analysis": ["dca"],
            "detect_overfitting": ["overfit"],
            "bootstrap_validate": ["bval"],
        },
    )

    # 20. sensitivity.py (12 fns + 5 classes)
    total += _auto_register(
        "sensitivity",
        "Sensitivity",
        aliases={
            "e_value_rr": ["ev_rr"],
            "e_value_or": ["ev_or"],
            "e_value_hr": ["ev_hr"],
            "e_value_d": ["ev_d"],
            "rosenbaum_bounds": ["rosen"],
            "specification_curve": ["spec_curve"],
            "manski_bounds": ["manski"],
            "omitted_variable_bias": ["ovb"],
        },
    )

    # 21. tables_pub.py (11 fns + 1 class)
    total += _auto_register(
        "tables_pub",
        "Tables",
        aliases={
            "table1": ["t1"],
            "regression_table": ["reg_tab"],
            "odds_ratio_table": ["or_tab"],
            "hazard_ratio_table": ["hr_tab"],
            "model_comparison_table": ["mc_tab"],
            "treatment_effect_table": ["te_tab"],
        },
    )

    # 22. sampling.py (9 fns)
    total += _auto_register(
        "sampling",
        "Sampling",
        aliases={
            "simple_random_sample": ["srs"],
            "stratified_sample": ["strat_samp"],
            "cluster_sample": ["clust_samp"],
            "pps_sample": ["pps"],
        },
    )

    # 23. survey.py (7 fns + 1 class)
    total += _auto_register(
        "survey",
        "Survey",
        aliases={
            "horvitz_thompson_total": ["ht_total"],
            "hajek_mean": ["hajek"],
            "complex_survey_glm": ["svy_glm"],
            "calibration_weights": ["cal_wts"],
        },
    )

    # 24. data.py (10 fns + 1 class)
    total += _auto_register(
        "data",
        "Data",
        aliases={
            "load_dataset": ["ld"],
            "list_datasets": ["lsd"],
            "dataset_info": ["dinfo"],
        },
    )

    # 25. ml.py (2 fns)
    total += _auto_register(
        "ml",
        "ML",
        aliases={
            "apply_smote": ["smote"],
            "eval_robustness": ["robust"],
        },
    )

    # 26. ebac.py (2 fns)
    total += _auto_register(
        "ebac",
        "eBAC",
        aliases={
            "calculate_ebac": ["ebac"],
            "is_over_legal_limit": ["over_limit"],
        },
    )

    # 27. investigation.py (3 fns)
    total += _auto_register("investigation", "Investigation")

    # 28. psymet.py (11 fns — short names, ≤6 chars)
    total += _auto_register(
        "psymet",
        "Psychometrics",
        aliases={
            "crba": ["alpha", "cronbach"],
            "mcdo": ["omega", "mcdonald"],
            "itcor": ["itc"],
            "adel": ["aid"],
            "crel": ["cr"],
            "ave": [],
            "kmo": [],
            "bart": ["bartlett"],
            "paran": ["parallel", "horn"],
            "splhf": ["splhalf"],
            "idisc": ["disc"],
        },
    )

    # 29. otis.py (6 fns — correctional/sociolegal)
    total += _auto_register(
        "otis",
        "Correctional",
        aliases={
            "rplace": ["regplace"],
            "astcmb": ["alerts"],
            "volat": ["vol"],
            "rctrnd": ["trends"],
            "otdesc": ["otdescr"],
            "otdml": ["otirm"],
        },
    )

    return total


# ---------------------------------------------------------------------------
# Wrangling commands (pandas wrappers)
# ---------------------------------------------------------------------------


def _register_wrangling() -> int:
    """Register ~80 pandas wrangling commands."""
    count = 0

    wrangling_commands = {
        # Reshaping
        "wgl_merge": ("Merge two DataFrames", "merge(df1, df2, on, how)"),
        "wgl_join": ("Join DataFrames on index", "join(df1, df2, how)"),
        "wgl_concat": ("Concatenate DataFrames", "concat(dfs, axis)"),
        "wgl_pivot": ("Pivot table", "pivot(df, index, columns, values)"),
        "wgl_pivot_table": ("Pivot table with aggregation", "pivot_table(df, index, columns, values, aggfunc)"),
        "wgl_melt": ("Unpivot wide to long", "melt(df, id_vars, value_vars)"),
        "wgl_stack": ("Stack columns to rows", "stack(df)"),
        "wgl_unstack": ("Unstack rows to columns", "unstack(df)"),
        "wgl_explode": ("Explode list column", "explode(df, col)"),
        "wgl_transpose": ("Transpose DataFrame", "transpose(df)"),
        # Filtering
        "wgl_query": ("Query with expression", "query(df, expr)"),
        "wgl_filter_rows": ("Filter rows by condition", "filter_rows(df, col, op, val)"),
        "wgl_filter_between": ("Filter between values", "filter_between(df, col, lo, hi)"),
        "wgl_filter_isin": ("Filter where column in list", "filter_isin(df, col, vals)"),
        "wgl_filter_notna": ("Filter non-null rows", "filter_notna(df, col)"),
        "wgl_filter_contains": ("Filter by string contains", "filter_contains(df, col, pattern)"),
        "wgl_nlargest": ("Top N rows by column", "nlargest(df, n, col)"),
        "wgl_nsmallest": ("Bottom N rows by column", "nsmallest(df, n, col)"),
        "wgl_drop_duplicates": ("Remove duplicate rows", "drop_duplicates(df, cols)"),
        "wgl_sample_frac": ("Random fraction of rows", "sample_frac(df, frac)"),
        # Column operations
        "wgl_select": ("Select columns", "select(df, cols)"),
        "wgl_drop": ("Drop columns", "drop(df, cols)"),
        "wgl_rename": ("Rename columns", "rename(df, old, new)"),
        "wgl_reorder": ("Reorder columns", "reorder(df, cols)"),
        "wgl_astype": ("Cast column type", "astype(df, col, dtype)"),
        "wgl_to_numeric": ("Convert to numeric", "to_numeric(df, col)"),
        "wgl_to_datetime": ("Convert to datetime", "to_datetime(df, col)"),
        "wgl_to_category": ("Convert to category", "to_category(df, col)"),
        "wgl_fillna": ("Fill missing values", "fillna(df, col, value)"),
        "wgl_dropna": ("Drop rows with NaN", "dropna(df, subset)"),
        "wgl_clip": ("Clip values to range", "clip(df, col, lo, hi)"),
        "wgl_replace": ("Replace values", "replace(df, col, old, new)"),
        "wgl_bin": ("Bin continuous variable", "bin(df, col, bins)"),
        "wgl_qcut": ("Quantile-based binning", "qcut(df, col, q)"),
        "wgl_cut": ("Fixed-width binning", "cut(df, col, bins)"),
        "wgl_get_dummies": ("One-hot encode", "get_dummies(df, col)"),
        "wgl_apply": ("Apply function to column", "apply(df, col, func)"),
        "wgl_map": ("Map values", "map(df, col, mapping)"),
        "wgl_assign": ("Create new column", "assign(df, name, expr)"),
        # Aggregation
        "wgl_groupby": ("Group by and aggregate", "groupby(df, by, col, func)"),
        "wgl_agg": ("Multiple aggregations", "agg(df, col, funcs)"),
        "wgl_resample": ("Time-based resampling", "resample(df, freq, func)"),
        "wgl_rolling": ("Rolling window", "rolling(df, col, window, func)"),
        "wgl_expanding": ("Expanding window", "expanding(df, col, func)"),
        "wgl_ewm": ("Exponential weighted", "ewm(df, col, span, func)"),
        "wgl_cumsum": ("Cumulative sum", "cumsum(df, col)"),
        "wgl_cummax": ("Cumulative max", "cummax(df, col)"),
        "wgl_cummin": ("Cumulative min", "cummin(df, col)"),
        "wgl_cumprod": ("Cumulative product", "cumprod(df, col)"),
        "wgl_rank": ("Rank values", "rank(df, col, method)"),
        "wgl_pct_change": ("Percent change", "pct_change(df, col)"),
        "wgl_diff": ("Difference", "diff(df, col, periods)"),
        # Sorting
        "wgl_sort": ("Sort by column", "sort(df, col, ascending)"),
        "wgl_sort_index": ("Sort by index", "sort_index(df)"),
        # String operations
        "wgl_str_upper": ("Uppercase", "str_upper(df, col)"),
        "wgl_str_lower": ("Lowercase", "str_lower(df, col)"),
        "wgl_str_strip": ("Strip whitespace", "str_strip(df, col)"),
        "wgl_str_replace": ("String replace", "str_replace(df, col, old, new)"),
        "wgl_str_contains": ("String contains check", "str_contains(df, col, pat)"),
        "wgl_str_extract": ("Regex extract", "str_extract(df, col, pat)"),
        "wgl_str_split": ("Split string", "str_split(df, col, sep)"),
        "wgl_str_len": ("String length", "str_len(df, col)"),
        "wgl_str_pad": ("Pad string", "str_pad(df, col, width, side)"),
        "wgl_str_slice": ("Slice string", "str_slice(df, col, start, stop)"),
        # Date operations
        "wgl_dt_year": ("Extract year", "dt_year(df, col)"),
        "wgl_dt_month": ("Extract month", "dt_month(df, col)"),
        "wgl_dt_day": ("Extract day", "dt_day(df, col)"),
        "wgl_dt_dayofweek": ("Day of week", "dt_dayofweek(df, col)"),
        "wgl_dt_hour": ("Extract hour", "dt_hour(df, col)"),
        "wgl_dt_quarter": ("Extract quarter", "dt_quarter(df, col)"),
        "wgl_dt_diff": ("Date difference", "dt_diff(df, col1, col2)"),
        "wgl_dt_floor": ("Floor date", "dt_floor(df, col, freq)"),
        "wgl_dt_ceil": ("Ceil date", "dt_ceil(df, col, freq)"),
        # Misc
        "wgl_info": ("DataFrame info summary", "info(df)"),
        "wgl_memory": ("Memory usage", "memory(df)"),
        "wgl_dtypes": ("Column data types", "dtypes(df)"),
        "wgl_nunique": ("Count unique per column", "nunique(df)"),
        "wgl_value_counts": ("Value counts", "value_counts(df, col)"),
        "wgl_describe_col": ("Describe single column", "describe_col(df, col)"),
        "wgl_corr": ("Correlation matrix", "corr(df)"),
        "wgl_cov": ("Covariance matrix", "cov(df)"),
    }

    for cmd_name, (desc, usage) in wrangling_commands.items():

        def _make_wgl_stat(cn: str, d: str) -> Callable:
            def handler(parts: list[str], log: Any, store: Callable) -> None:
                log.write(f"[bold]{cn}[/bold]: {d}")
                log.write(f"[dim]Usage: {cn} <csv> <args>[/dim]")
                log.write("[dim]Load data first, then call from REPL for full functionality.[/dim]")

            return handler

        def _make_wgl_repl(cn: str) -> Callable:
            def handler(*args: Any, **kwargs: Any) -> Any:
                import pandas as pd

                op = cn.replace("wgl_", "")
                if args and isinstance(args[0], pd.DataFrame):
                    df = args[0]
                    rest = args[1:]
                    if hasattr(df, op):
                        return getattr(df, op)(*rest, **kwargs)
                raise ValueError(f"Usage: {cn}(df, ...) — pass a DataFrame as first arg")

            handler.__name__ = cn
            handler.__doc__ = f"Pandas {cn.replace('wgl_', '')} wrapper"
            return handler

        register(
            StatCommand(
                name=cmd_name,
                category="Wrangling",
                usage=usage,
                description=desc,
                handler_stat=_make_wgl_stat(cmd_name, desc),
                handler_repl=_make_wgl_repl(cmd_name),
                aliases=[cmd_name.replace("wgl_", "")],  # e.g. wgl_merge -> merge alias
            )
        )
        count += 1

    return count


# ---------------------------------------------------------------------------
# Descriptive statistics (pandas-based)
# ---------------------------------------------------------------------------


def _register_descriptive() -> int:
    """Register ~60 descriptive statistics commands."""
    count = 0

    desc_commands = {
        "desc_mean": ("Arithmetic mean", "mean(df, col)"),
        "desc_median": ("Median", "median(df, col)"),
        "desc_mode": ("Mode", "mode(df, col)"),
        "desc_std": ("Standard deviation", "std(df, col)"),
        "desc_var": ("Variance", "var(df, col)"),
        "desc_sem": ("Standard error of mean", "sem(df, col)"),
        "desc_min": ("Minimum value", "min(df, col)"),
        "desc_max": ("Maximum value", "max(df, col)"),
        "desc_range": ("Range (max - min)", "range(df, col)"),
        "desc_iqr": ("Interquartile range", "iqr(df, col)"),
        "desc_mad": ("Median absolute deviation", "mad(df, col)"),
        "desc_skew": ("Skewness", "skew(df, col)"),
        "desc_kurtosis": ("Kurtosis", "kurtosis(df, col)"),
        "desc_sum": ("Sum", "sum(df, col)"),
        "desc_count": ("Count non-null", "count(df, col)"),
        "desc_quantile": ("Quantile", "quantile(df, col, q)"),
        "desc_percentile": ("Percentile", "percentile(df, col, p)"),
        "desc_q25": ("25th percentile", "q25(df, col)"),
        "desc_q50": ("50th percentile (median)", "q50(df, col)"),
        "desc_q75": ("75th percentile", "q75(df, col)"),
        "desc_q90": ("90th percentile", "q90(df, col)"),
        "desc_q95": ("95th percentile", "q95(df, col)"),
        "desc_q99": ("99th percentile", "q99(df, col)"),
        "desc_ci_mean": ("95% CI for mean", "ci_mean(df, col)"),
        "desc_ci_proportion": ("95% CI for proportion", "ci_proportion(df, col)"),
        "desc_cv": ("Coefficient of variation", "cv(df, col)"),
        "desc_zscore": ("Z-scores", "zscore(df, col)"),
        "desc_winsorize": ("Winsorize outliers", "winsorize(df, col, limits)"),
        "desc_trimmed_mean": ("Trimmed mean", "trimmed_mean(df, col, pct)"),
        "desc_geometric_mean": ("Geometric mean", "geometric_mean(df, col)"),
        "desc_harmonic_mean": ("Harmonic mean", "harmonic_mean(df, col)"),
        "desc_weighted_mean": ("Weighted mean", "weighted_mean(df, col, weights)"),
        "desc_weighted_std": ("Weighted standard deviation", "weighted_std(df, col, weights)"),
        "desc_entropy": ("Shannon entropy", "entropy(df, col)"),
        "desc_gini": ("Gini coefficient", "gini(df, col)"),
        "desc_five_num": ("Five-number summary", "five_num(df, col)"),
        "desc_seven_num": ("Seven-number summary", "seven_num(df, col)"),
        "desc_outliers_iqr": ("IQR outlier detection", "outliers_iqr(df, col)"),
        "desc_outliers_zscore": ("Z-score outlier detection", "outliers_zscore(df, col, threshold)"),
        "desc_grubbs": ("Grubbs test for outliers", "grubbs(df, col)"),
        "desc_biweight_mean": ("Biweight mean (robust)", "biweight_mean(df, col)"),
        "desc_biweight_std": ("Biweight std (robust)", "biweight_std(df, col)"),
        "desc_crosstab": ("Cross tabulation", "crosstab(df, col1, col2)"),
        "desc_frequency_table": ("Frequency table", "frequency_table(df, col)"),
        "desc_contingency_table": ("Contingency table", "contingency_table(df, col1, col2)"),
        "desc_summary": ("R-style summary", "summary(df)"),
        "desc_profile": ("Full dataset profile", "profile(df)"),
        "desc_completeness": ("Completeness report", "completeness(df)"),
        "desc_n_unique": ("Unique count per column", "n_unique(df)"),
        "desc_top_values": ("Top N values", "top_values(df, col, n)"),
        "desc_bottom_values": ("Bottom N values", "bottom_values(df, col, n)"),
        "desc_histogram_data": ("Histogram bin data", "histogram_data(df, col, bins)"),
        "desc_boxplot_data": ("Boxplot statistics", "boxplot_data(df, col)"),
        "desc_ecdf": ("Empirical CDF data", "ecdf(df, col)"),
        "desc_kernel_density": ("Kernel density estimate", "kernel_density(df, col)"),
        "desc_moments": ("First 4 statistical moments", "moments(df, col)"),
        "desc_cumulants": ("Cumulants", "cumulants(df, col)"),
        "desc_robust_scale": ("Robust scale estimate (MAD)", "robust_scale(df, col)"),
        "desc_l_moments": ("L-moments", "l_moments(df, col)"),
        "desc_describe_all": ("Describe all numeric columns", "describe_all(df)"),
    }

    for cmd_name, (desc, usage) in desc_commands.items():

        def _make_desc_stat(cn: str, d: str) -> Callable:
            def handler(parts: list[str], log: Any, store: Callable) -> None:
                log.write(f"[bold]{cn}[/bold]: {d}")
                log.write("[dim]Load data first in REPL for full functionality.[/dim]")

            return handler

        def _make_desc_repl(cn: str) -> Callable:
            def handler(*args: Any, **kwargs: Any) -> Any:
                import numpy as np
                import pandas as pd

                op = cn.replace("desc_", "")

                if not args:
                    raise ValueError(f"Usage: {cn}(df, col)")

                data = args[0]
                col = args[1] if len(args) > 1 else None

                if isinstance(data, pd.DataFrame) and col:
                    series = data[col].dropna()
                elif isinstance(data, pd.Series):
                    series = data.dropna()
                elif isinstance(data, (list, np.ndarray)):
                    series = pd.Series(data).dropna()
                else:
                    raise ValueError(f"Pass DataFrame+col, Series, or array to {cn}")

                # Dispatch common operations
                simple_ops = {
                    "mean": lambda s: s.mean(),
                    "median": lambda s: s.median(),
                    "mode": lambda s: s.mode().tolist(),
                    "std": lambda s: s.std(),
                    "var": lambda s: s.var(),
                    "sem": lambda s: s.sem(),
                    "min": lambda s: s.min(),
                    "max": lambda s: s.max(),
                    "range": lambda s: s.max() - s.min(),
                    "iqr": lambda s: s.quantile(0.75) - s.quantile(0.25),
                    "mad": lambda s: (s - s.median()).abs().median(),
                    "skew": lambda s: s.skew(),
                    "kurtosis": lambda s: s.kurtosis(),
                    "sum": lambda s: s.sum(),
                    "count": lambda s: s.count(),
                    "q25": lambda s: s.quantile(0.25),
                    "q50": lambda s: s.quantile(0.50),
                    "q75": lambda s: s.quantile(0.75),
                    "q90": lambda s: s.quantile(0.90),
                    "q95": lambda s: s.quantile(0.95),
                    "q99": lambda s: s.quantile(0.99),
                    "cv": lambda s: s.std() / s.mean() if s.mean() != 0 else float("nan"),
                    "entropy": lambda s: (
                        -(s.value_counts(normalize=True) * np.log2(s.value_counts(normalize=True))).sum()
                    ),
                    "n_unique": lambda s: s.nunique(),
                }

                if op in simple_ops:
                    return simple_ops[op](series)

                # More complex operations
                if op == "five_num":
                    return {
                        "min": series.min(),
                        "Q1": series.quantile(0.25),
                        "median": series.median(),
                        "Q3": series.quantile(0.75),
                        "max": series.max(),
                    }
                if op == "ci_mean":
                    from scipy import stats

                    n = len(series)
                    m = series.mean()
                    se = series.sem()
                    t_val = stats.t.ppf(0.975, n - 1)
                    return {"mean": m, "ci_lower": m - t_val * se, "ci_upper": m + t_val * se}
                if op == "zscore":
                    return (series - series.mean()) / series.std()
                if op == "outliers_iqr":
                    q1, q3 = series.quantile(0.25), series.quantile(0.75)
                    iqr = q3 - q1
                    return series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)]
                if op == "quantile":
                    q = args[2] if len(args) > 2 else 0.5
                    return series.quantile(float(q))
                if op == "trimmed_mean":
                    from scipy import stats

                    pct = float(args[2]) if len(args) > 2 else 0.1
                    return stats.trim_mean(series.values, pct)
                if op == "geometric_mean":
                    from scipy import stats

                    return stats.gmean(series[series > 0].values)
                if op == "harmonic_mean":
                    from scipy import stats

                    return stats.hmean(series[series > 0].values)

                return series.describe()

            handler.__name__ = cn
            handler.__doc__ = f"Descriptive: {desc}"
            return handler

        register(
            StatCommand(
                name=cmd_name,
                category="Descriptive",
                usage=usage,
                description=desc,
                handler_stat=_make_desc_stat(cmd_name, desc),
                handler_repl=_make_desc_repl(cmd_name),
                aliases=[cmd_name.replace("desc_", "")],
            )
        )
        count += 1

    return count


# ---------------------------------------------------------------------------
# R bridge commands
# ---------------------------------------------------------------------------


def _register_r_bridge() -> int:
    """Register ~80 R bridge commands that call R via subprocess."""
    count = 0

    r_commands = {
        # Statistical tests
        "r_ttest": ("R t.test()", "r_ttest(x, mu)"),
        "r_ttest2": ("R two-sample t.test()", "r_ttest2(x, y)"),
        "r_paired_t": ("R paired t.test()", "r_paired_t(x, y)"),
        "r_wilcox": ("R wilcox.test()", "r_wilcox(x, y)"),
        "r_anova": ("R aov()", "r_anova(formula, data)"),
        "r_kruskal": ("R kruskal.test()", "r_kruskal(formula, data)"),
        "r_chisq": ("R chisq.test()", "r_chisq(x, y)"),
        "r_fisher": ("R fisher.test()", "r_fisher(x, y)"),
        "r_shapiro": ("R shapiro.test()", "r_shapiro(x)"),
        "r_ks": ("R ks.test()", "r_ks(x, y)"),
        "r_cor": ("R cor.test()", "r_cor(x, y, method)"),
        "r_mcnemar": ("R mcnemar.test()", "r_mcnemar(x)"),
        "r_bartlett": ("R bartlett.test()", "r_bartlett(formula, data)"),
        "r_levene": ("R car::leveneTest()", "r_levene(formula, data)"),
        "r_friedman": ("R friedman.test()", "r_friedman(formula, data)"),
        # Regression
        "r_lm": ("R lm()", "r_lm(formula, data)"),
        "r_glm": ("R glm()", "r_glm(formula, data, family)"),
        "r_logistic": ("R glm(family=binomial)", "r_logistic(formula, data)"),
        "r_poisson": ("R glm(family=poisson)", "r_poisson(formula, data)"),
        "r_nls": ("R nls()", "r_nls(formula, data, start)"),
        "r_lme": ("R nlme::lme()", "r_lme(formula, data, random)"),
        "r_lmer": ("R lme4::lmer()", "r_lmer(formula, data)"),
        "r_glmer": ("R lme4::glmer()", "r_glmer(formula, data, family)"),
        "r_gam": ("R mgcv::gam()", "r_gam(formula, data)"),
        "r_quantreg": ("R quantreg::rq()", "r_quantreg(formula, data, tau)"),
        "r_robust": ("R MASS::rlm()", "r_robust(formula, data)"),
        # Survival
        "r_surv": ("R survival::Surv()", "r_surv(time, event)"),
        "r_survfit": ("R survival::survfit()", "r_survfit(formula, data)"),
        "r_coxph": ("R survival::coxph()", "r_coxph(formula, data)"),
        "r_survdiff": ("R survival::survdiff()", "r_survdiff(formula, data)"),
        "r_aft": ("R survival::survreg()", "r_aft(formula, data, dist)"),
        # Causal
        "r_matchit": ("R MatchIt::matchit()", "r_matchit(formula, data, method)"),
        "r_ipw": ("R ipw::ipwpoint()", "r_ipw(formula, data)"),
        "r_aipw": ("R AIPW::AIPW()", "r_aipw(y, a, w)"),
        "r_dml": ("R DoubleML::DoubleMLPLR()", "r_dml(data, y, d, x)"),
        "r_irm": ("R DoubleML::DoubleMLIRM()", "r_irm(data, y, d, x)"),
        "r_didR": ("R did::att_gt()", "r_didR(formula, data, gname, tname)"),
        "r_rdrobust": ("R rdrobust::rdrobust()", "r_rdrobust(y, x, c)"),
        "r_ivreg": ("R ivreg::ivreg()", "r_ivreg(formula, data)"),
        "r_synth": ("R Synth::synth()", "r_synth(data)"),
        # Multiple testing
        "r_p_adjust": ("R p.adjust()", "r_p_adjust(p, method)"),
        # Diagnostics
        "r_vif": ("R car::vif()", "r_vif(model)"),
        "r_durbinwatson": ("R car::durbinWatsonTest()", "r_durbinwatson(model)"),
        "r_bptest": ("R lmtest::bptest()", "r_bptest(model)"),
        "r_resettest": ("R lmtest::resettest()", "r_resettest(model)"),
        # Effect sizes
        "r_cohens_d": ("R effectsize::cohens_d()", "r_cohens_d(x, y)"),
        "r_hedges_g": ("R effectsize::hedges_g()", "r_hedges_g(x, y)"),
        "r_eta_sq": ("R effectsize::eta_squared()", "r_eta_sq(model)"),
        "r_cramers_v": ("R effectsize::cramers_v()", "r_cramers_v(x)"),
        # Power
        "r_power_t": ("R pwr::pwr.t.test()", "r_power_t(d, n, sig, power)"),
        "r_power_anova": ("R pwr::pwr.anova.test()", "r_power_anova(k, n, f, sig)"),
        "r_power_chisq": ("R pwr::pwr.chisq.test()", "r_power_chisq(w, N, df, sig)"),
        "r_power_prop": ("R pwr::pwr.2p.test()", "r_power_prop(h, n, sig)"),
        # Tables
        "r_table1": ("R tableone::CreateTableOne()", "r_table1(data, vars, strata)"),
        "r_gtsummary": ("R gtsummary::tbl_summary()", "r_gtsummary(data, by)"),
        "r_stargazer": ("R stargazer::stargazer()", "r_stargazer(model)"),
        # Missing
        "r_mice": ("R mice::mice()", "r_mice(data, m, method)"),
        "r_amelia": ("R Amelia::amelia()", "r_amelia(data, m)"),
        "r_naniar": ("R naniar::vis_miss()", "r_naniar(data)"),
        # Visualization
        "r_ggplot": ("R ggplot2 wrapper", "r_ggplot(data, aes)"),
        "r_ggsurvplot": ("R survminer::ggsurvplot()", "r_ggsurvplot(fit, data)"),
        "r_forestplot": ("R forestplot::forestplot()", "r_forestplot(data)"),
        # Weights
        "r_svydesign": ("R survey::svydesign()", "r_svydesign(ids, weights, data)"),
        "r_svymean": ("R survey::svymean()", "r_svymean(formula, design)"),
        "r_svytotal": ("R survey::svytotal()", "r_svytotal(formula, design)"),
        "r_svyglm": ("R survey::svyglm()", "r_svyglm(formula, design, family)"),
        "r_calibrate": ("R survey::calibrate()", "r_calibrate(design, formula, pop)"),
        "r_rake": ("R survey::rake()", "r_rake(design, sample, population)"),
        # Bootstrap
        "r_boot": ("R boot::boot()", "r_boot(data, statistic, R)"),
        "r_boot_ci": ("R boot::boot.ci()", "r_boot_ci(boot_obj, type)"),
        # Meta-analysis
        "r_metafor": ("R metafor::rma()", "r_metafor(yi, vi, method)"),
        "r_meta_bin": ("R meta::metabin()", "r_meta_bin(event_e, n_e, event_c, n_c)"),
        # Misc
        "r_summary": ("R summary()", "r_summary(obj)"),
        "r_str": ("R str()", "r_str(obj)"),
        "r_head": ("R head()", "r_head(data, n)"),
        "r_tail": ("R tail()", "r_tail(data, n)"),
        "r_dim": ("R dim()", "r_dim(data)"),
        "r_names": ("R names()", "r_names(data)"),
        "r_class": ("R class()", "r_class(obj)"),
    }

    for cmd_name, (desc, usage) in r_commands.items():

        def _make_r_stat(cn: str, d: str) -> Callable:
            def handler(parts: list[str], log: Any, store: Callable) -> None:
                import shutil
                import subprocess

                if not shutil.which("Rscript"):
                    log.write("[red]R not found. Install R to use R bridge commands.[/red]")
                    return
                r_func = cn.replace("r_", "")
                args = parts[1:]
                _args_joined = ", ".join(repr(a) for a in args)
                r_code = f'cat("{r_func} called with args:", paste({_args_joined}, collapse=", "), "\\n")'
                try:
                    out = subprocess.run(
                        ["Rscript", "-e", r_code],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if out.stdout:
                        log.write(out.stdout.strip())
                    if out.stderr:
                        log.write(f"[dim]{out.stderr.strip()}[/dim]")
                except Exception as e:
                    log.write(f"[red]R error: {e}[/red]")

            return handler

        def _make_r_repl(cn: str) -> Callable:
            def handler(*args: Any, **kwargs: Any) -> str:
                import shutil
                import subprocess

                if not shutil.which("Rscript"):
                    raise RuntimeError("R not found")
                r_func = cn.replace("r_", "")
                str_args = ", ".join(repr(a) for a in args)
                r_code = f"result <- {r_func}({str_args}); print(result)"
                out = subprocess.run(
                    ["Rscript", "-e", r_code],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if out.returncode != 0:
                    raise RuntimeError(out.stderr)
                return out.stdout

            handler.__name__ = cn
            handler.__doc__ = f"R bridge: {desc}"
            return handler

        register(
            StatCommand(
                name=cmd_name,
                category="R Bridge",
                usage=usage,
                description=desc,
                handler_stat=_make_r_stat(cmd_name, desc),
                handler_repl=_make_r_repl(cmd_name),
                aliases=[],
                is_r_bridge=True,
            )
        )
        count += 1

    return count


# ---------------------------------------------------------------------------
# Compound commands (multi-step workflows)
# ---------------------------------------------------------------------------


def _register_compound() -> int:
    """Register ~120 compound workflow commands."""
    count = 0

    compound_defs = {
        # Survival workflows
        "full_survival": ("Full survival analysis: KM + Cox + logrank", ["kaplan_meier", "cox_ph", "logrank_test"]),
        "survival_diagnostics": (
            "Survival diagnostics: PH test + residuals",
            ["test_ph_assumption", "schoenfeld_residuals", "cox_snell_residuals"],
        ),
        "parametric_survival": (
            "Compare parametric survival models",
            ["weibull_model", "lognormal_model", "loglogistic_model", "compare_parametric_models"],
        ),
        "competing_risks": ("Competing risks analysis", ["cumulative_incidence_function", "fine_gray_weights"]),
        # Causal workflows
        "full_causal": (
            "Full causal: PS + IPW + ATE + sensitivity",
            ["compute_propensity_scores", "calculate_ipw_weights", "estimate_ate", "e_value"],
        ),
        "full_aipw": (
            "AIPW workflow: PS + AIPW + diagnostics",
            ["compute_propensity_scores", "estimate_aipw", "balance_diagnostics"],
        ),
        "full_dml": ("DML workflow: PLR + IRM", ["estimate_plr", "estimate_irm"]),
        "full_matching": (
            "Full matching: PS + NN match + balance + ATT",
            ["estimate_propensity_score", "match_nearest_neighbor", "balance_diagnostics", "estimate_att_matched"],
        ),
        "ps_analysis": (
            "Propensity score analysis pipeline",
            ["estimate_propensity_score", "common_support", "trim_propensity_scores", "overlap_diagnostics"],
        ),
        "dr_estimation": (
            "Doubly robust estimation",
            ["estimate_propensity_score", "doubly_robust_matching", "abadie_imbens_se"],
        ),
        # DiD workflows
        "full_did": (
            "Full DiD: 2x2 + parallel trends + event study",
            ["did_2x2", "test_parallel_trends", "event_study"],
        ),
        "staggered_analysis": (
            "Staggered DiD: group-time ATT + aggregation + bacon",
            ["group_time_att", "aggregate_gt_att", "bacon_decomposition"],
        ),
        "did_robustness": (
            "DiD robustness: placebo tests + sensitivity",
            ["placebo_test_time", "placebo_test_outcome", "did_sensitivity_analysis"],
        ),
        # RDD workflows
        "full_rdd": ("Full RDD: sharp + density test + bandwidth", ["sharp_rdd", "mccrary_test", "bandwidth_cct"]),
        "rdd_robustness": (
            "RDD robustness: bandwidth sensitivity + placebo + donut",
            ["bandwidth_sensitivity", "placebo_cutoff_test", "donut_rdd"],
        ),
        # IV workflows
        "full_iv": (
            "Full IV: TSLS + diagnostics + overid",
            ["tsls", "first_stage_diagnostics", "sargan_test", "hausman_test"],
        ),
        "iv_robustness": ("IV robustness: LIML + JIVE + Anderson-Rubin", ["liml", "jive", "anderson_rubin_test"]),
        # Matching workflows
        "compare_matching": (
            "Compare matching methods",
            ["match_nearest_neighbor", "match_exact", "match_mahalanobis", "match_optimal_pair"],
        ),
        "matching_sensitivity": (
            "Matching sensitivity: Rosenbaum bounds + balance",
            ["rosenbaum_bounds", "balance_diagnostics", "love_plot_data"],
        ),
        # Missing data workflows
        "full_missing": ("Full missing: profile + MCAR test + MICE", ["missing_profile", "littles_mcar_test", "mice"]),
        "imputation_check": (
            "Imputation diagnostics + pooling",
            ["imputation_diagnostics", "pool_mice_estimates", "rubins_rules"],
        ),
        # Effect size workflows
        "full_effects": (
            "All effect sizes for two groups",
            ["cohens_d", "hedges_g", "glass_delta", "cles", "rank_biserial_correlation"],
        ),
        "meta_analysis": (
            "Meta-analysis workflow: FE + RE + heterogeneity",
            ["fixed_effects_meta", "random_effects_meta", "i_squared", "prediction_interval"],
        ),
        # Testing workflows
        "full_normality": ("Full normality battery", ["normality_battery"]),
        "full_variance": ("Full variance equality battery", ["variance_equality_battery"]),
        "nonparametric_suite": (
            "Non-parametric test suite",
            ["mann_whitney_u", "wilcoxon_signed_rank", "kruskal_wallis", "friedman_test"],
        ),
        "parametric_suite": (
            "Parametric test suite",
            ["one_sample_ttest", "two_sample_ttest", "paired_ttest", "one_way_anova"],
        ),
        "association_suite": (
            "Association test suite",
            ["pearson_correlation", "spearman_correlation", "kendall_correlation", "chi2_independence"],
        ),
        # Diagnostic workflows
        "full_diagnostics": (
            "Full model diagnostics",
            ["compute_residuals", "compute_vif", "compute_influence", "hosmer_lemeshow_test"],
        ),
        "collinearity_check": ("Collinearity assessment", ["compute_vif", "collinearity_diagnostics"]),
        "specification_tests": (
            "Specification test battery",
            ["ramsey_reset_test", "link_test", "wald_test", "score_test"],
        ),
        # Sensitivity workflows
        "full_sensitivity": (
            "Full sensitivity: E-value + Rosenbaum + OVB",
            ["e_value_rr", "rosenbaum_bounds", "omitted_variable_bias"],
        ),
        "bias_assessment": (
            "Bias assessment battery",
            ["manski_bounds", "bias_adjusted_estimate", "probabilistic_bias_analysis"],
        ),
        # Reporting workflows
        "full_report": (
            "Generate complete report",
            [
                "generate_introduction",
                "generate_methods",
                "generate_results",
                "generate_discussion",
                "generate_limitations",
            ],
        ),
        "strobe_audit": ("STROBE compliance audit", ["check_strobe_compliance", "audit_statistical_reporting"]),
        "reproducibility_check": ("Reproducibility assessment", ["create_reproducibility_manifest"]),
        # Multiple testing workflows
        "p_correction_suite": (
            "All p-value corrections",
            ["bonferroni", "holm", "hochberg", "benjamini_hochberg", "storey_q"],
        ),
        "combined_tests": (
            "Combined probability tests",
            ["fisher_combined", "stouffer_combined", "tippett_combined", "simes_combined"],
        ),
        # Validation workflows
        "full_validation": (
            "Full validation: CV + calibration + discrimination",
            ["cross_validate", "assess_calibration", "assess_discrimination"],
        ),
        "overfitting_check": ("Overfitting assessment", ["detect_overfitting", "bootstrap_validate"]),
        # Bootstrap workflows
        "bootstrap_suite": ("Bootstrap methods comparison", ["bootstrap", "parametric_bootstrap", "wild_bootstrap"]),
        "resampling_suite": (
            "Resampling method comparison",
            ["bootstrap", "jackknife", "permutation_test", "cross_validate"],
        ),
        # Weight workflows
        "full_weighting": (
            "Full weighting: design + calibrate + diagnostics",
            ["compute_design_weights", "calibrate_to_totals", "weight_diagnostics"],
        ),
        "weight_assessment": (
            "Weight quality assessment",
            ["weight_diagnostics", "detect_extreme_weights", "effective_sample_size"],
        ),
        # Table workflows
        "publication_tables": ("All publication tables", ["table1", "regression_table", "model_comparison_table"]),
        # Survey workflows
        "survey_analysis": ("Full survey analysis", ["horvitz_thompson_total", "hajek_mean", "ratio_estimator"]),
        # Visualization workflows
        "diagnostic_plots": ("All diagnostic plots", ["qq_plot", "residual_diagnostic_plots", "influence_plot"]),
        "causal_plots": ("Causal inference plots", ["propensity_score_plot", "balance_plot", "forest_plot"]),
        "survival_plots": ("Survival analysis plots", ["kaplan_meier_plot", "forest_plot"]),
        # Export workflows
        "export_all_formats": ("Export to all formats", ["df_to_latex", "df_to_html", "df_to_markdown", "df_to_docx"]),
        "export_manuscript": (
            "Export manuscript bundle",
            ["export_results_bundle", "generate_citation", "create_reproducibility_manifest"],
        ),
        # Distribution workflows
        "distribution_suite": ("Distribution function suite", ["dnorm", "pnorm", "qnorm", "rnorm"]),
        # Power analysis workflows
        "power_suite": ("Power analysis suite", ["power_t_test", "power_prop_test", "power_anova"]),
        # Sampling workflows
        "sampling_comparison": (
            "Compare sampling methods",
            ["simple_random_sample", "stratified_sample", "cluster_sample"],
        ),
        # Cross-method comparison
        "compare_ate_methods": (
            "Compare ATE estimation methods",
            ["estimate_ate", "estimate_aipw", "estimate_double_ml"],
        ),
        "compare_did_methods": ("Compare DiD estimators", ["did_2x2", "staggered_did", "did_doubly_robust"]),
        "compare_rdd_methods": (
            "Compare RDD estimators",
            ["sharp_rdd", "rdd_bias_corrected", "rdd_local_randomisation"],
        ),
        "compare_iv_methods": ("Compare IV estimators", ["tsls", "liml", "gmm_iv"]),
        "compare_matching_methods": (
            "Compare matching methods",
            ["match_nearest_neighbor", "match_mahalanobis", "match_cem", "entropy_balance"],
        ),
        "compare_imputation": (
            "Compare imputation methods",
            ["impute_mean", "impute_median", "impute_regression", "mice"],
        ),
        "compare_survival": ("Compare survival models", ["kaplan_meier", "cox_ph", "weibull_model", "aft_weibull"]),
        "compare_bootstrap": (
            "Compare bootstrap methods",
            ["bootstrap", "parametric_bootstrap", "wild_bootstrap", "block_bootstrap"],
        ),
        # Quick analysis templates
        "quick_eda": ("Quick exploratory data analysis", ["missing_profile", "normality_battery"]),
        "quick_ttest": ("Quick t-test with effect size", ["two_sample_ttest", "cohens_d"]),
        "quick_anova": ("Quick ANOVA with effect size", ["one_way_anova", "eta_squared"]),
        "quick_chi2": ("Quick chi-square with effect size", ["chi2_independence", "cramers_v"]),
        "quick_regression": ("Quick regression with diagnostics", ["compute_vif", "compute_residuals"]),
        "quick_logistic": ("Quick logistic with diagnostics", ["hosmer_lemeshow_test", "compute_goodness_of_fit"]),
        "quick_survival_km": ("Quick KM analysis", ["kaplan_meier", "logrank_test"]),
        "quick_cox": ("Quick Cox analysis", ["cox_ph", "test_ph_assumption"]),
        "quick_ipw": (
            "Quick IPW analysis",
            ["compute_propensity_scores", "calculate_ipw_weights", "effective_sample_size"],
        ),
        "quick_match": (
            "Quick matching analysis",
            ["estimate_propensity_score", "match_nearest_neighbor", "balance_diagnostics"],
        ),
        "quick_did": ("Quick DiD analysis", ["did_2x2", "test_parallel_trends"]),
        "quick_rdd": ("Quick RDD analysis", ["sharp_rdd", "mccrary_test"]),
        "quick_iv": ("Quick IV analysis", ["tsls", "first_stage_diagnostics"]),
        "quick_meta": ("Quick meta-analysis", ["random_effects_meta", "i_squared"]),
        "quick_power": ("Quick power analysis for t-test", ["power_t_test"]),
        "quick_missing": ("Quick missing data assessment", ["missing_profile", "littles_mcar_test"]),
        "quick_sensitivity": ("Quick sensitivity analysis", ["e_value_rr", "rosenbaum_bounds"]),
        "quick_table1": ("Quick Table 1", ["table1"]),
        "quick_forest": ("Quick forest plot", ["forest_plot"]),
        "quick_validation": ("Quick model validation", ["cross_validate", "assess_calibration"]),
        # Comprehensive analysis templates
        "rct_analysis": (
            "Full RCT analysis pipeline",
            ["table1", "two_sample_ttest", "cohens_d", "check_strobe_compliance"],
        ),
        "observational_study": (
            "Full observational study pipeline",
            [
                "table1",
                "compute_propensity_scores",
                "match_nearest_neighbor",
                "estimate_att_matched",
                "rosenbaum_bounds",
            ],
        ),
        "cohort_study": ("Full cohort study pipeline", ["table1", "kaplan_meier", "cox_ph", "test_ph_assumption"]),
        "case_control": ("Case-control study pipeline", ["table1", "odds_ratio", "logistic_regression"]),
        "cross_sectional": ("Cross-sectional study pipeline", ["table1", "chi2_independence", "pearson_correlation"]),
        "time_series_causal": ("Time series causal pipeline", ["did_2x2", "event_study", "test_parallel_trends"]),
        "policy_evaluation": ("Policy evaluation pipeline", ["did_2x2", "sharp_rdd", "tsls"]),
        "drug_trial": (
            "Drug trial analysis pipeline",
            ["table1", "kaplan_meier", "cox_ph", "restricted_mean_survival_time"],
        ),
        "epidemiological": ("Epidemiological analysis pipeline", ["table1", "odds_ratio", "risk_ratio", "e_value_rr"]),
        "survey_study": ("Survey study pipeline", ["horvitz_thompson_total", "hajek_mean", "complex_survey_glm"]),
        # Data quality
        "data_quality": ("Data quality assessment", ["missing_profile", "validate_schema", "score_data_quality"]),
        "data_cleaning": ("Data cleaning checklist", ["missing_profile"]),
        # Reproducibility
        "reproducibility_bundle": (
            "Create reproducibility bundle",
            ["create_reproducibility_manifest", "generate_citation"],
        ),
    }

    for cmd_name, (desc, steps) in compound_defs.items():

        def _make_compound_stat(cn: str, d: str, st: list) -> Callable:
            def handler(parts: list[str], log: Any, store: Callable) -> None:
                log.write(f"\n[bold magenta]Compound: {cn}[/bold magenta]")
                log.write(f"[dim]{d}[/dim]")
                log.write(f"[dim]Steps: {' → '.join(st)}[/dim]\n")
                for step_name in st:
                    step_cmd = resolve(step_name)
                    if step_cmd:
                        log.write(f"[bold]── {step_name} ──[/bold]")
                        try:
                            step_cmd.handler_stat(parts, log, store)
                        except Exception as e:
                            log.write(f"[yellow]Skipped {step_name}: {e}[/yellow]")
                    else:
                        log.write(f"[dim]Step {step_name} not found in registry[/dim]")
                log.write(f"\n[green]Compound analysis '{cn}' complete.[/green]")

            return handler

        def _make_compound_repl(cn: str, d: str, st: list) -> Callable:
            def handler(*args: Any, **kwargs: Any) -> dict:
                results = {}
                for step_name in st:
                    step_cmd = resolve(step_name)
                    if step_cmd:
                        try:
                            results[step_name] = step_cmd.handler_repl(*args, **kwargs)
                        except Exception as e:
                            results[step_name] = f"Error: {e}"
                return results

            handler.__name__ = cn
            handler.__doc__ = f"Compound: {d}. Steps: {', '.join(st)}"
            return handler

        register(
            StatCommand(
                name=cmd_name,
                category="Compound",
                usage=f"{cmd_name} <csv> <args>",
                description=desc,
                handler_stat=_make_compound_stat(cmd_name, desc, steps),
                handler_repl=_make_compound_repl(cmd_name, desc, steps),
                aliases=[],
                is_compound=True,
            )
        )
        count += 1

    return count


# ---------------------------------------------------------------------------
# Initialize — register everything at import time
# ---------------------------------------------------------------------------


def _register_fn_shortcuts() -> int:
    """Register moirais.fn short-name aliases for individual function files.

    Each fn/ file exports a function with a short name (max 7 chars).
    This registers them as aliases pointing to the canonical command
    already registered from the backend module, OR as new commands
    if not already present.
    """
    count = 0

    # Mapping: short_name -> (fn_module, func_name, category, description)
    FN_MAP = {
        # eBAC
        "ebac": ("fn.ebac", "calculate_ebac", "eBAC", "Compute estimated blood alcohol concentration"),
        "legal": ("fn.legal", "is_over_legal_limit", "eBAC", "Check if eBAC exceeds legal limit"),
        # ML
        "robust": ("fn.robust", "eval_robustness", "ML", "Random forest robustness check"),
        "smote": ("fn.smote", "apply_smote", "ML", "SMOTE oversampling for class balance"),
        # Psychometrics
        "crba": ("fn.crba", "crba", "Psychometrics", "Cronbach's alpha reliability"),
        "mcdo": ("fn.mcdo", "mcdo", "Psychometrics", "McDonald's omega total + hierarchical"),
        "itcor": ("fn.itcor", "itcor", "Psychometrics", "Corrected item-total correlations"),
        "adel": ("fn.adel", "adel", "Psychometrics", "Alpha if item deleted"),
        "crel": ("fn.crel", "crel", "Psychometrics", "Composite reliability from loadings"),
        "ave": ("fn.ave", "ave", "Psychometrics", "Average variance extracted"),
        "kmo": ("fn.kmo", "kmo", "Psychometrics", "Kaiser-Meyer-Olkin sampling adequacy"),
        "bart": ("fn.bart", "bart", "Psychometrics", "Bartlett's test of sphericity"),
        "paran": ("fn.paran", "paran", "Psychometrics", "Horn's parallel analysis"),
        "splhf": ("fn.splhf", "splhf", "Psychometrics", "Spearman-Brown split-half reliability"),
        "idisc": ("fn.idisc", "idisc", "Psychometrics", "Item discrimination index"),
        # OTIS
        "rpl": ("fn.rpl", "rplace", "OTIS", "Regional placement analysis"),
        "astc": ("fn.astc", "astcmb", "OTIS", "Alert-state combination encoding"),
        "vol": ("fn.vol", "volat", "OTIS", "Regional volatility metric"),
        "rct": ("fn.rct", "rctrnd", "OTIS", "Restrictive confinement trends"),
        "otd": ("fn.otd", "otdesc", "OTIS", "OTIS descriptive statistics"),
        "oml": ("fn.oml", "otdml", "OTIS", "DML treatment effect on OTIS data"),
        # Inference — distributions
        "dnorm": ("fn.dnorm", "dnorm", "Distribution", "Normal density"),
        "pnorm": ("fn.pnorm", "pnorm", "Distribution", "Normal CDF"),
        "qnorm": ("fn.qnorm", "qnorm", "Distribution", "Normal quantile"),
        "rnorm": ("fn.rnorm", "rnorm", "Distribution", "Normal random variates"),
        "dt": ("fn.dt", "dt", "Distribution", "Student t density"),
        "pt": ("fn.pt", "pt", "Distribution", "Student t CDF"),
        "qt": ("fn.qt", "qt", "Distribution", "Student t quantile"),
        "dchsq": ("fn.dchsq", "dchisq", "Distribution", "Chi-square density"),
        "pchsq": ("fn.pchsq", "pchisq", "Distribution", "Chi-square CDF"),
        "qchsq": ("fn.qchsq", "qchisq", "Distribution", "Chi-square quantile"),
        "dbnm": ("fn.dbnm", "dbinom", "Distribution", "Binomial PMF"),
        "pbnm": ("fn.pbnm", "pbinom", "Distribution", "Binomial CDF"),
        "qbnm": ("fn.qbnm", "qbinom", "Distribution", "Binomial quantile"),
        "rbnm": ("fn.rbnm", "rbinom", "Distribution", "Binomial random variates"),
        "dpoi": ("fn.dpoi", "dpois", "Distribution", "Poisson PMF"),
        "ppoi": ("fn.ppoi", "ppois", "Distribution", "Poisson CDF"),
        "qpoi": ("fn.qpoi", "qpois", "Distribution", "Poisson quantile"),
        "rpoi": ("fn.rpoi", "rpois", "Distribution", "Poisson random variates"),
        "dbet": ("fn.dbet", "dbeta", "Distribution", "Beta density"),
        "pbet": ("fn.pbet", "pbeta", "Distribution", "Beta CDF"),
        "qbet": ("fn.qbet", "qbeta", "Distribution", "Beta quantile"),
        "dgam": ("fn.dgam", "dgamma", "Distribution", "Gamma density"),
        "pgam": ("fn.pgam", "pgamma", "Distribution", "Gamma CDF"),
        "dunf": ("fn.dunf", "dunif", "Distribution", "Uniform density"),
        "punf": ("fn.punf", "punif", "Distribution", "Uniform CDF"),
        "runf": ("fn.runf", "runif", "Distribution", "Uniform random variates"),
        # Inference — tests
        "t2smp": ("fn.t2smp", "two_sample_t_test", "Test", "Two-sample t-test"),
        "t1smp": ("fn.t1smp", "one_sample_t_test", "Test", "One-sample t-test"),
        "tpair": ("fn.tpair", "paired_t_test", "Test", "Paired t-test"),
        "chisq": ("fn.chisq", "chi_square_test", "Test", "Chi-square test"),
        "fisher": ("fn.fisher", "fisher_exact_test", "Test", "Fisher exact test"),
        "anova": ("fn.anova", "anova_one_way", "Test", "One-way ANOVA"),
        "kw": ("fn.kw", "kruskal_wallis_test", "Test", "Kruskal-Wallis test"),
        "mw": ("fn.mw", "mann_whitney_test", "Test", "Mann-Whitney U test"),
        "wilcox": ("fn.wilcox", "wilcoxon_signed_rank_test", "Test", "Wilcoxon signed-rank"),
        "sw": ("fn.sw", "shapiro_wilk_test", "Test", "Shapiro-Wilk normality test"),
        "levene": ("fn.levene", "levene_test", "Test", "Levene homogeneity test"),
        # Inference — CIs
        "prop_ci": ("fn.prop_ci", "proportion_ci", "CI", "Proportion confidence interval"),
        "rr_ci": ("fn.rr_ci", "rate_ratio_ci", "CI", "Rate ratio CI"),
        "or_ci": ("fn.or_ci", "odds_ratio_ci", "CI", "Odds ratio CI"),
        "rsk_ci": ("fn.rsk_ci", "risk_ratio_ci", "CI", "Risk ratio CI"),
        "rd_ci": ("fn.rd_ci", "risk_difference_ci", "CI", "Risk difference CI"),
        # Inference — effect sizes
        "d": ("fn.d", "cohens_d", "EffectSize", "Cohen's d"),
        "g": ("fn.g", "hedges_g", "EffectSize", "Hedges' g"),
        "eta2": ("fn.eta2", "eta_squared", "EffectSize", "Eta-squared"),
        "omega2": ("fn.omega2", "omega_squared", "EffectSize", "Omega-squared"),
        "cramv": ("fn.cramv", "cramers_v", "EffectSize", "Cramer's V"),
        "phi": ("fn.phi", "phi_coefficient", "EffectSize", "Phi coefficient"),
        "pbr": ("fn.pbr", "point_biserial_r", "EffectSize", "Point-biserial r"),
        "tau": ("fn.tau", "kendall_tau", "EffectSize", "Kendall's tau"),
        "rho": ("fn.rho", "spearman_rho", "EffectSize", "Spearman's rho"),
        # Inference — power
        "pwr_t": ("fn.pwr_t", "power_t_test", "Power", "Power for t-test"),
        "pwr_p": ("fn.pwr_p", "power_prop_test", "Power", "Power for proportions"),
        "pwr_av": ("fn.pwr_av", "power_anova", "Power", "Power for ANOVA"),
        "n_logit": ("fn.n_logit", "sample_size_logistic", "Power", "Sample size for logistic"),
        # Inference — other
        "i_pwr": ("fn.i_pwr", "calculate_interaction_power", "Power", "Interaction power"),
        "boot": ("fn.boot", "bootstrap_ci", "Inference", "Bootstrap confidence interval"),
    }

    for short, (fn_mod, func_name, cat, desc) in FN_MAP.items():
        # Skip if already registered as alias or command
        if short in COMMAND_REGISTRY or short in ALIAS_MAP:
            continue

        register(
            StatCommand(
                name=short,
                category=cat,
                usage=f"{short} ...",
                description=desc,
                handler_stat=_make_stat_handler(fn_mod, func_name),
                handler_repl=_make_repl_handler(fn_mod, func_name),
                aliases=[func_name] if func_name != short else [],
                module=fn_mod,
            )
        )
        count += 1

    return count


def _init_registry() -> None:
    """Register all commands. Called once at module import."""
    total = 0
    total += _register_all_backend_modules()
    total += _register_fn_shortcuts()
    total += _register_wrangling()
    total += _register_descriptive()
    total += _register_r_bridge()
    total += _register_compound()


# Run registration
_init_registry()
