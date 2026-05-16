"""Top-level package for the MORIE Python interface.

All heavy submodule imports are lazy via PEP 562 __getattr__: an
attribute is loaded the first time it is referenced, not at
`import morie` time.  This keeps `import morie` cold-startup under
a second even when the install has optional heavy deps (DoubleML,
statsmodels, sklearn, lxml, etc.) -- previously these were eagerly
imported in try/except blocks at the bottom of the package's
__init__.py, costing ~2 minutes on a cold cache.

For minimal envs without an optional submodule's dependencies, the
lazy loader returns AttributeError (as if the symbol were never
exposed), matching the previous try/except-pass behaviour.
"""

try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
    try:
        __version__ = _pkg_version("morie")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"
    del _pkg_version, PackageNotFoundError
except ImportError:
    __version__ = "0.0.0+unknown"

# name -> submodule (".cpads", ".causal", ...).  Generated from the
# previous eager try-import block at v0.3.0.  Edit this dict to add
# new top-level exports instead of adding more try-imports.
_LAZY_EXPORTS = {
    'AnovaOneWayResult': 'mrm_design',
    'CPADS_REQUIRED_VARIABLES': 'cpads',
    'CausalDesignResult': 'mrm_design',
    'ColumnProfile': 'dataset',
    'DatasetProfile': 'dataset',
    'DatasetRegistry': 'data',
    'Factorial2kResult': 'mrm_design',
    'GPL_COMPATIBLE_LICENSES': '_license_check',
    'LISAResult': 'mrm_lisa',
    'LongitudinalSimSpec': 'longitudinal_sim',
    'MeasurementLevel': 'dataset',
    'SIU_INDEX_URL': 'siu_fetch',
    'ScanCluster': 'mrm_kulldorff',
    'TPS_LAYER_URLS': 'tps_fetch',
    'TwoTreatmentResult': 'mrm_design',
    'agent_available': 'perseus',
    'ask': 'llm',
    'ask_multi': 'llm',
    'ask_percy': 'perseus',
    'bootstrap_sample': 'sampling',
    'build_morie_context': 'llm',
    'build_prompt': 'perseus',
    'calculate_ebac': 'ebac',
    'calculate_ipw_weights': 'causal',
    'canonicalize_cpads_frame': 'cpads',
    'check_datasets': 'data',
    'dataset_recommendation': 'data',
    'check_plugin_license': '_license_check',
    'cluster_sample': 'sampling',
    'compare_nested_logistic_models': 'investigation',
    'compute_design_weights': 'sampling',
    'compute_propensity_scores': 'causal',
    'cpads_contract': 'cpads',
    'design_effect': 'sampling',
    'detect_available_provider': 'llm',
    'effective_sample_size': 'causal',
    'estimate_atc': 'causal',
    'estimate_ate': 'effects',
    'estimate_att': 'causal',
    'estimate_cate': 'causal',
    'estimate_gate': 'causal',
    'estimate_irm': 'causal',
    'estimate_late': 'causal',
    'execute_pipeline as _execute_pipeline': 'runner',
    'fairness_average_odds_difference': 'fairness.metrics',
    'fairness_bias_amplification': 'fairness.metrics',
    'fairness_demographic_parity': 'fairness.metrics',
    'fairness_disparate_impact': 'fairness.metrics',
    'fairness_equalized_odds': 'fairness.metrics',
    'fairness_gini': 'fairness.metrics',
    'CityProfile': 'fairness.cityprofile',
    'apply_profile': 'fairness.cityprofile',
    'get_city': 'fairness.cityprofile',
    'list_cities': 'fairness.cityprofile',
    'register_city': 'fairness.cityprofile',
    'predpol_aggregate_areas': 'fairness.predpol',
    'predpol_calibration_audit': 'fairness.predpol',
    'predpol_score_disparity': 'fairness.predpol',
    'predpol_temporal_audit': 'fairness.temporal',
    'noisy_or_detection': 'fairness.simulation',
    'simulate_biased_crime_data': 'fairness.simulation',
    'SpatialGAN': 'fairness.gan',
    'CTGANDebiaser': 'fairness.gan',
    'xai_permutation_importance': 'fairness.xai',
    'xai_partial_dependence': 'fairness.xai',
    'xai_ale': 'fairness.xai',
    'xai_ceteris_paribus': 'fairness.xai',
    'xai_shap_values': 'fairness.xai',
    'fetch_siu_cases': 'siu_fetch',
    'fetch_tps_category': 'tps_fetch',
    'generate_ar_coefficients': 'longitudinal_sim',
    'generate_var_coefficients': 'longitudinal_sim',
    'infer_measurement_level': 'dataset',
    'inspect_output': 'inspector',
    'is_over_legal_limit': 'ebac',
    'jackknife_estimate': 'sampling',
    'launch_tui as _launch_tui': 'tui',
    'list_modules': 'modules',
    'list_tps_categories': 'tps_fetch',
    'load_dataset': 'dataset',
    'morie_license_metadata': '_license_check',
    'mrm_anova_bonferroni': 'mrm_doe',
    'mrm_anova_oneway': 'mrm_design',
    'mrm_anova_power': 'mrm_doe',
    'mrm_assumptions_check': 'mrm_diagnostics',
    'mrm_causal_design': 'mrm_design',
    'mrm_check_balancing': 'mrm_diagnostics',
    'mrm_check_overlap': 'mrm_diagnostics',
    'mrm_classify_mandela': 'mrm_otis',
    'mrm_clt_demo': 'mrm_mathstats',
    'mrm_factorial_2k': 'mrm_design',
    'mrm_fractional_factorial': 'mrm_doe',
    'mrm_graeco_latin': 'mrm_doe',
    'mrm_latin_square': 'mrm_doe',
    'mrm_mc_power': 'mrm_doe',
    'mrm_median_causal_effect': 'mrm_diagnostics',
    'mrm_oneprop_test': 'mrm_mathstats',
    'mrm_otis_mandela_spectrum': 'mrm_mandela_spectrum',
    'mrm_otis_mortification_cooccurrence': 'mrm_otis',
    'mrm_otis_placement_concentration': 'mrm_otis',
    'mrm_otis_region_locality': 'mrm_otis',
    'mrm_otis_seg_duration_km': 'mrm_otis',
    'mrm_perm_block': 'mrm_doe',
    'mrm_pit': 'mrm_mathstats',
    'mrm_qq_plot': 'mrm_mathstats',
    'mrm_random_latin': 'mrm_doe',
    'mrm_rcbd': 'mrm_doe',
    'mrm_response_surface': 'mrm_doe',
    'mrm_siu_case_to_decision_km': 'mrm_siu',
    'mrm_siu_outcome_classifier': 'mrm_siu',
    'mrm_siu_per_service_rate': 'mrm_siu',
    'mrm_standardised_difference': 'mrm_diagnostics',
    'mrm_tps_kulldorff_scan': 'mrm_kulldorff',
    'mrm_tps_levy_scaling': 'mrm_tps',
    'mrm_tps_lisa': 'mrm_lisa',
    'mrm_tps_load_hawkes_refit': 'mrm_tps',
    'mrm_tps_moran_clustering': 'mrm_tps',
    'mrm_tps_neighbourhood_recurrence_km': 'mrm_tps',
    'mrm_tps_polygon_moran_per_year': 'mrm_lisa',
    'mrm_two_treatment_test': 'mrm_design',
    'mrm_twoprop_test': 'mrm_mathstats',
    'mrm_var_test': 'mrm_mathstats',
    'mvn_with_covariance': 'longitudinal_sim',
    'pps_sample': 'sampling',
    'profile_dataset': 'dataset',
    'run_chat_repl as _run_chat_repl': 'chat',
    'run_ebac_selection_ipw_analysis': 'causal',
    'run_module': 'modules',
    'run_modules': 'modules',
    'run_power_design_module': 'modules',
    'run_propensity_ipw_analysis': 'causal',
    'run_treatment_effects_analysis': 'investigation',
    'run_weighted_logistic_analysis': 'investigation',
    'simple_random_sample': 'sampling',
    'simulate_longitudinal_panel': 'longitudinal_sim',
    'siu_cache_path': 'siu_fetch',
    'stratified_sample': 'sampling',
    'suggest_analysis_plan': 'dataset',
    'sync_rng': 'longitudinal_sim',
    'validate_cpads_frame': 'cpads',
    'verify_statistical_output': 'inspector',
}

def __getattr__(name):
    """PEP 562 lazy loader for top-level morie.* exports."""
    if name in _LAZY_EXPORTS:
        from importlib import import_module
        try:
            mod = import_module("." + _LAZY_EXPORTS[name], package=__name__)
            obj = getattr(mod, name)
            globals()[name] = obj
            return obj
        except (ImportError, AttributeError):
            # Match the previous try/except-pass behaviour: pretend the
            # symbol doesn't exist when its dependencies are absent.
            raise AttributeError(
                f"morie has no attribute {name!r} "
                f"(optional submodule {_LAZY_EXPORTS[name]!r} could not be loaded)"
            )
    raise AttributeError(f"module 'morie' has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_EXPORTS) + ["__version__", "__getattr__", "__dir__", "load_sample"])


def load_sample(name: str):
    """Load a bundled reference sample CSV by name.

    Available samples: 'otis_b01', 'otis_b09', 'otis_c11', 'tps_assault'.
    The OTIS samples are taken from the public Ontario Data Catalogue
    release; the TPS sample is taken from Toronto Police Open Data.
    """
    import pandas as pd
    from pathlib import Path
    here = Path(__file__).parent / "data" / "samples"
    files = {
        "otis_b01": "otis_b01_sample.csv",
        "otis_b09": "otis_b09_sample.csv",
        "otis_c11": "otis_c11_sample.csv",
        "tps_assault": "tps_assault_sample.csv",
    }
    if name not in files:
        raise KeyError(f"Unknown sample {name!r}; choices: {list(files)}")
    return pd.read_csv(here / files[name])


# Aliases that the old __init__.py exposed (not generated from a from-import)
# Lazily resolved via __getattr__ above when the underlying perseus submodule
# is available; these names route to the same callables.
_LAZY_EXPORTS.setdefault("assistant_available", "perseus")
_LAZY_EXPORTS.setdefault("ask_morie_assistant", "perseus")
_LAZY_EXPORTS.setdefault("build_assistant_prompt", "perseus")


__all__ = [
    "ColumnProfile",
    "DatasetProfile",
    "DatasetRegistry",
    "CPADS_REQUIRED_VARIABLES",
    "MeasurementLevel",
    "ask",
    "agent_available",
    "ask_percy",
    "ask_morie_assistant",
    "ask_multi",
    "assistant_available",
    "build_prompt",
    "bootstrap_sample",
    "build_assistant_prompt",
    "build_morie_context",
    "calculate_ebac",
    "calculate_ipw_weights",
    "canonicalize_cpads_frame",
    "cluster_sample",
    "compare_nested_logistic_models",
    "compute_design_weights",
    "compute_propensity_scores",
    "cpads_contract",
    "design_effect",
    "detect_available_provider",
    "estimate_ate",
    "estimate_att",
    "estimate_atc",
    "estimate_cate",
    "estimate_gate",
    "estimate_irm",
    "estimate_late",
    "effective_sample_size",
    "execute_pipeline",
    "inspect_output",
    "launch_tui",
    "infer_measurement_level",
    "is_over_legal_limit",
    "jackknife_estimate",
    "list_modules",
    "load_dataset",
    "pps_sample",
    "profile_dataset",
    "run_module",
    "run_modules",
    "run_power_design_module",
    "run_chat_repl",
    "run_ebac_selection_ipw_analysis",
    "run_propensity_ipw_analysis",
    "run_treatment_effects_analysis",
    "run_weighted_logistic_analysis",
    "simple_random_sample",
    "stratified_sample",
    "suggest_analysis_plan",
    "validate_cpads_frame",
    "verify_statistical_output",
    # Fairness & disparity-audit subsystem (morie.fairness)
    "fairness_disparate_impact",
    "fairness_demographic_parity",
    "fairness_equalized_odds",
    "fairness_average_odds_difference",
    "fairness_gini",
    "fairness_bias_amplification",
    "predpol_calibration_audit",
    "predpol_aggregate_areas",
    "predpol_score_disparity",
    "predpol_temporal_audit",
    "noisy_or_detection",
    "simulate_biased_crime_data",
    "SpatialGAN",
    "CTGANDebiaser",
    "xai_permutation_importance",
    "xai_partial_dependence",
    "xai_ale",
    "xai_ceteris_paribus",
    "xai_shap_values",
    "CityProfile",
    "register_city",
    "get_city",
    "list_cities",
    "apply_profile",
    # MRM-framework empirical analyses
    "mrm_classify_mandela",
    "mrm_otis_placement_concentration",
    "mrm_otis_seg_duration_km",
    "mrm_otis_mortification_cooccurrence",
    "mrm_otis_region_locality",
    "mrm_tps_levy_scaling",
    "mrm_tps_moran_clustering",
    "mrm_tps_neighbourhood_recurrence_km",
    "mrm_tps_load_hawkes_refit",
    "mrm_siu_case_to_decision_km",
    "mrm_siu_per_service_rate",
    "mrm_siu_outcome_classifier",
    # Dataset fetchers (on-demand download/scrape)
    "TPS_LAYER_URLS",
    "fetch_tps_category",
    "list_tps_categories",
    "SIU_INDEX_URL",
    "fetch_siu_cases",
    "siu_cache_path",
    "mrm_otis_mandela_spectrum",
    "mrm_tps_lisa",
    "mrm_tps_polygon_moran_per_year",
    "LISAResult",
    "mrm_two_treatment_test",
    "mrm_anova_oneway",
    "mrm_factorial_2k",
    "mrm_causal_design",
    # Tier 1 diagnostics
    "mrm_standardised_difference",
    "mrm_check_balancing",
    "mrm_check_overlap",
    "mrm_median_causal_effect",
    "mrm_assumptions_check",
    # Tier 2 math-stats
    "mrm_oneprop_test",
    "mrm_twoprop_test",
    "mrm_var_test",
    "mrm_qq_plot",
    "mrm_clt_demo",
    "mrm_pit",
    # Tier 3 DOE
    "mrm_anova_bonferroni",
    "mrm_rcbd",
    "mrm_latin_square",
    "mrm_graeco_latin",
    "mrm_fractional_factorial",
    "mrm_response_surface",
    "mrm_anova_power",
    "mrm_mc_power",
    "mrm_perm_block",
    "mrm_random_latin",
    "load_sample",
    "sync_rng",
    "generate_ar_coefficients",
    "generate_var_coefficients",
    "mvn_with_covariance",
    "simulate_longitudinal_panel",
    "LongitudinalSimSpec",
    "mrm_tps_kulldorff_scan",
    "ScanCluster",
    "GPL_COMPATIBLE_LICENSES",
    "check_datasets",
    "dataset_recommendation",
    "check_plugin_license",
    "morie_license_metadata",

    # New modules (import via morie.viz, morie.tables_pub, etc.)
    "viz",
    "tables_pub",
    "validation",
    "export",
]


# Fail-silent, daily-cached check for a newer morie release on PyPI.
# The import hot path only reads a small cache file; any network
# request runs in a background daemon thread.  This is how an existing
# user on an older version is told a new one exists.  Opt out with the
# MORIE_NO_UPDATE_CHECK environment variable.
try:
    from ._update_check import maybe_notify as _maybe_notify
    _maybe_notify(__version__)
    del _maybe_notify
except Exception:
    pass
