"""Top-level package for the MORIE Python interface.

All heavy imports are guarded so that lightweight submodules (morie.quant,
morie.fn, morie.ebac, etc.) can be imported in minimal environments (e.g.
autoresearch venv) without requiring sklearn, httpx, textual, or other
optional dependencies.
"""

__version__ = "0.1.15"

# --- Guarded eager imports — fail gracefully in minimal envs ---
# In a full morie install these all succeed and populate the namespace.
# In a minimal env (only numpy/scipy), they silently skip.

try:
    from .cpads import CPADS_REQUIRED_VARIABLES, canonicalize_cpads_frame, cpads_contract, validate_cpads_frame
except ImportError:
    pass

try:
    from .causal import (
        calculate_ipw_weights,
        compute_propensity_scores,
        effective_sample_size,
        estimate_atc,
        estimate_att,
        estimate_cate,
        estimate_gate,
        estimate_irm,
        estimate_late,
        run_ebac_selection_ipw_analysis,
        run_propensity_ipw_analysis,
    )
except ImportError:
    pass

try:
    from .data import DatasetRegistry
except ImportError:
    pass

try:
    from .dataset import (
        ColumnProfile,
        DatasetProfile,
        MeasurementLevel,
        infer_measurement_level,
        load_dataset,
        profile_dataset,
        suggest_analysis_plan,
    )
except ImportError:
    pass

try:
    from .ebac import calculate_ebac, is_over_legal_limit
except ImportError:
    pass

try:
    from .mrm_otis import (
        mrm_otis_placement_concentration,
        mrm_otis_seg_duration_km,
        mrm_otis_mortification_cooccurrence,
        mrm_otis_region_locality,
    )
except ImportError:
    pass

try:
    from .mrm_tps import (
        mrm_tps_levy_scaling,
        mrm_tps_moran_clustering,
        mrm_tps_neighbourhood_recurrence_km,
        mrm_tps_load_hawkes_refit,
    )
except ImportError:
    pass

try:
    from .mrm_siu import (
        mrm_siu_case_to_decision_km,
        mrm_siu_per_service_rate,
        mrm_siu_outcome_classifier,
    )
except ImportError:
    pass

try:
    from .tps_fetch import (
        TPS_LAYER_URLS,
        fetch_tps_category,
        list_tps_categories,
    )
except ImportError:
    pass

try:
    from .siu_fetch import (
        SIU_INDEX_URL,
        fetch_siu_cases,
        siu_cache_path,
    )
except ImportError:
    pass



try:
    from .longitudinal_sim import (
        sync_rng,
        generate_ar_coefficients,
        generate_var_coefficients,
        mvn_with_covariance,
        simulate_longitudinal_panel,
        LongitudinalSimSpec,
    )
except ImportError:
    pass

try:
    from .mrm_kulldorff import mrm_tps_kulldorff_scan, ScanCluster
except ImportError:
    pass

try:
    from ._license_check import (
        GPL_COMPATIBLE_LICENSES,
        check_plugin_license,
        morie_license_metadata,
    )
except ImportError:
    pass

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

try:
    from .effects import estimate_ate
except ImportError:
    pass

try:
    from .investigation import (
        compare_nested_logistic_models,
        run_treatment_effects_analysis,
        run_weighted_logistic_analysis,
    )
except ImportError:
    pass

try:
    from .inspector import inspect_output, verify_statistical_output
except ImportError:
    pass

try:
    from .modules import list_modules, run_module, run_modules, run_power_design_module
except ImportError:
    pass

try:
    from .sampling import (
        bootstrap_sample,
        cluster_sample,
        compute_design_weights,
        design_effect,
        jackknife_estimate,
        pps_sample,
        simple_random_sample,
        stratified_sample,
    )
except ImportError:
    pass

try:
    from .perseus import agent_available, ask_percy, build_prompt

    assistant_available = agent_available
    ask_morie_assistant = ask_percy
    build_assistant_prompt = build_prompt
    from .llm import ask, ask_multi, build_morie_context, detect_available_provider
except ImportError:
    pass

# --- New modules: viz, tables_pub, validation, export ---
# Lazy imports to avoid heavy matplotlib/sklearn load on basic usage.


def _lazy_viz():
    from . import viz as _viz

    return _viz


def _lazy_tables_pub():
    from . import tables_pub as _tables_pub

    return _tables_pub


def _lazy_validation():
    from . import validation as _validation

    return _validation


def _lazy_export():
    from . import export as _export

    return _export


def execute_pipeline(*args, **kwargs):
    """Lazily import the CLI runner helper to avoid module-execution warnings."""
    from .runner import execute_pipeline as _execute_pipeline

    return _execute_pipeline(*args, **kwargs)


def run_chat_repl(*args, **kwargs):
    """Lazily import the chat REPL."""
    from .chat import run_chat_repl as _run_chat_repl

    return _run_chat_repl(*args, **kwargs)


def launch_tui(*args, **kwargs):
    """Lazily import the TUI launcher."""
    from .tui import launch_tui as _launch_tui

    return _launch_tui(*args, **kwargs)


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
    # MRM-framework empirical analyses
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
    "check_plugin_license",
    "morie_license_metadata",

    # New modules (import via morie.viz, morie.tables_pub, etc.)
    "viz",
    "tables_pub",
    "validation",
    "export",
]
