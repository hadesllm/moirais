"""Top-level package for the MOIRAIS Python interface.

All heavy imports are guarded so that lightweight submodules (moirais.quant,
moirais.fn, moirais.ebac, etc.) can be imported in minimal environments (e.g.
autoresearch venv) without requiring sklearn, httpx, textual, or other
optional dependencies.
"""

__version__ = "0.1.2"

# --- Guarded eager imports — fail gracefully in minimal envs ---
# In a full moirais install these all succeed and populate the namespace.
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
    ask_moirais_assistant = ask_percy
    build_assistant_prompt = build_prompt
    from .llm import ask, ask_multi, build_moirais_context, detect_available_provider
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
    "ask_moirais_assistant",
    "ask_multi",
    "assistant_available",
    "build_prompt",
    "bootstrap_sample",
    "build_assistant_prompt",
    "build_moirais_context",
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
    # New modules (import via moirais.viz, moirais.tables_pub, etc.)
    "viz",
    "tables_pub",
    "validation",
    "export",
]
