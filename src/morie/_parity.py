"""General-purpose parity-review helpers for comparing legacy and modern codebases.

Three modes:
  1. General: ``compare_function_parity(source_dir, target_dir)`` — any two dirs.
  2. MORIE default: ``build_parity_matrix(epiml_root)`` — built-in epiml→morie mapping.
  3. TIDE: ``build_tide_parity()`` — Python TUI vs Go TIDE feature parity.

Example (general-purpose)::

    from morie._parity import compare_function_parity
    df = compare_function_parity("/path/to/old", "/path/to/new")

Example (custom)::

    script_map = {"old_auth.py": {"target_api": "...", "status": "done"}}
    df = build_parity_matrix("/path/to/old", script_map=script_map)
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ._migration import scan_directory


@dataclass(frozen=True)
class ParityReport:
    """Summary of function-level parity between two directory trees."""

    total_source_files: int
    total_target_files: int
    total_source_functions: int
    total_target_functions: int
    matched_functions: int
    source_only_functions: int
    target_only_functions: int
    parity_ratio: float


def compare_function_parity(
    source_dir: str | Path,
    target_dir: str | Path,
    *,
    extensions: set[str] | None = None,
) -> pd.DataFrame:
    """Compare function-level parity between two directory trees.

    :param source_dir: Legacy / source directory.
    :param target_dir: Modern / target directory.
    :param extensions: File extensions to scan.
    :return: DataFrame with per-function parity status.
    """
    source = scan_directory(source_dir, extensions=extensions)
    target = scan_directory(target_dir, extensions=extensions)

    source_fns: dict[str, str] = {}
    for path, info in source.items():
        for fn in info.get("functions", []):
            source_fns[fn] = path

    target_fns: dict[str, str] = {}
    for path, info in target.items():
        for fn in info.get("functions", []):
            target_fns[fn] = path

    rows: list[dict[str, Any]] = []
    all_fns = sorted(set(source_fns) | set(target_fns))
    for fn in all_fns:
        in_source = fn in source_fns
        in_target = fn in target_fns
        if in_source and in_target:
            status = "matched"
        elif in_source:
            status = "source_only"
        else:
            status = "target_only"
        rows.append(
            {
                "function": fn,
                "source_file": source_fns.get(fn, ""),
                "target_file": target_fns.get(fn, ""),
                "status": status,
            }
        )

    return pd.DataFrame(rows)


def summarize_function_parity(df: pd.DataFrame) -> ParityReport:
    """Summarize a function parity DataFrame into a ParityReport."""
    counts = df["status"].value_counts()
    matched = int(counts.get("matched", 0))
    source_only = int(counts.get("source_only", 0))
    target_only = int(counts.get("target_only", 0))
    total = matched + source_only
    return ParityReport(
        total_source_files=df["source_file"].nunique(),
        total_target_files=df["target_file"].nunique(),
        total_source_functions=matched + source_only,
        total_target_functions=matched + target_only,
        matched_functions=matched,
        source_only_functions=source_only,
        target_only_functions=target_only,
        parity_ratio=matched / total if total > 0 else 0.0,
    )


# ── MORIE-specific defaults ──────────────────────────────────────────────────

MORIE_SCRIPT_MAP = {
    "03_data_wrangling.R": {
        "target_morie_api": "morie run-module data-wrangling",
        "target_docs_page": "modules/investigation",
        "parity_status": "already present",
    },
    "04_descriptive_stats.R": {
        "target_morie_api": "morie run-module descriptive-statistics",
        "target_docs_page": "modules/investigation",
        "parity_status": "already present",
    },
    "04_distributions.R": {
        "target_morie_api": "morie run-module distribution-tests",
        "target_docs_page": "modules/investigation",
        "parity_status": "already present",
    },
    "05_frequentist.R": {
        "target_morie_api": "morie run-module frequentist-inference",
        "target_docs_page": "modules/investigation",
        "parity_status": "already present",
    },
    "05_bayesian.R": {
        "target_morie_api": "morie run-module bayesian-inference",
        "target_docs_page": "modules/investigation",
        "parity_status": "already present",
    },
    "05_power_design.R": {
        "target_morie_api": "morie run-module power-design",
        "target_docs_page": "modules/investigation",
        "parity_status": "already present",
    },
    "06_logistic.R": {
        "target_morie_api": "morie run-module logistic-models",
        "target_docs_page": "modules/investigation",
        "parity_status": "already present",
    },
    "06_model_comparison.R": {
        "target_morie_api": "morie run-module model-comparison",
        "target_docs_page": "modules/investigation",
        "parity_status": "already present",
    },
    "06_regression.R": {
        "target_morie_api": "morie run-module regression-models",
        "target_docs_page": "modules/investigation",
        "parity_status": "already present",
    },
    "07_propensity.R": {
        "target_morie_api": "morie run-module propensity-scores",
        "target_docs_page": "modules/investigation",
        "parity_status": "already present",
    },
    "07_causal_estimators.R": {
        "target_morie_api": "morie run-module causal-estimators",
        "target_docs_page": "modules/investigation",
        "parity_status": "already present",
    },
    "07_treatment_effects.R": {
        "target_morie_api": "morie run-module treatment-effects",
        "target_docs_page": "modules/investigation",
        "parity_status": "already present",
    },
    "07_dag.R": {
        "target_morie_api": "morie run-module dag-specification",
        "target_docs_page": "modules/investigation",
        "parity_status": "already present",
    },
    "07_meta_synthesis.R": {
        "target_morie_api": "morie run-module meta-synthesis",
        "target_docs_page": "modules/investigation",
        "parity_status": "scaffolded but statistically incomplete",
    },
    "08_figures.R": {
        "target_morie_api": "morie run-module figures",
        "target_docs_page": "modules/investigation",
        "parity_status": "scaffolded but statistically incomplete",
    },
    "08_tables.R": {
        "target_morie_api": "morie run-module tables",
        "target_docs_page": "modules/investigation",
        "parity_status": "scaffolded but statistically incomplete",
    },
    "09_report.R": {
        "target_morie_api": "morie run-module final-report",
        "target_docs_page": "modules/investigation",
        "parity_status": "already present",
    },
    "07_ebac.R": {
        "target_morie_api": "morie run-module ebac-core",
        "target_docs_page": "modules/ebac",
        "parity_status": "already present",
    },
    "07_ebac_ipw.R": {
        "target_morie_api": "morie run-module ebac-selection-adjustment-ipw",
        "target_docs_page": "modules/ebac",
        "parity_status": "already present",
    },
    "07_ebac_gender_smote_sensitivity.R": {
        "target_morie_api": "morie run-module ebac-gender-smote-sensitivity",
        "target_docs_page": "modules/ebac",
        "parity_status": "already present",
    },
    "07_ebac_integrations.R": {
        "target_morie_api": "morie run-module ebac-integrations",
        "target_docs_page": "modules/ebac",
        "parity_status": "already present",
    },
}

LEGACY_SCRIPT_MAP = MORIE_SCRIPT_MAP

MORIE_SURFACE_MAP = {
    "audit_public_outputs": "already present",
    "build_outputs_manifest": "already present",
    "default_synthetic_name_map": "already present",
    "default_workflow_map": "already present",
    "epiml_paths": "migrated as morie_paths",
    "find_project_root": "already present",
    "generate_synthetic_data": "already present",
    "read_outputs_manifest": "already present",
    "run_pipeline": "already present",
    "run_workflow_step": "already present",
    "summarize_output_audit": "already present",
    "validate_outputs_manifest": "already present",
    "write_synthetic_data": "already present",
}

PACKAGE_SURFACE_MAP = MORIE_SURFACE_MAP


def _current_r_exports(repo_root: Path, namespace_path: Path | None = None) -> set[str]:
    if namespace_path is None:
        namespace_path = repo_root / "r-package" / "morie" / "NAMESPACE"
    if not namespace_path.exists():
        return set()
    exports = set()
    for line in namespace_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export(") and line.endswith(")"):
            exports.add(line[len("export(") : -1])
    return exports


def _output_dir_candidates(epiml_root: Path) -> list[Path]:
    return [
        epiml_root / "project" / "data" / "public" / "outputs",
        epiml_root / "project" / "_site" / "data" / "public" / "outputs",
        epiml_root / "data" / "public" / "outputs",
        epiml_root / "six" / "outputs",
    ]


def _iter_unique_files(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _current_output_files(repo_root: Path) -> set[str]:
    output_root = repo_root / "data" / "public" / "outputs"
    if not output_root.exists():
        return set()
    return {str(path.relative_to(output_root)).replace("\\", "/") for path in output_root.rglob("*") if path.is_file()}


def _published_output_status(relative_name: str, path: Path, current_outputs: set[str]) -> str:
    if relative_name in current_outputs or path.name in current_outputs:
        return "already present"
    if path.suffix.lower() in {".pdf", ".png"}:
        return "scaffolded but statistically incomplete"
    return "missing and must be migrated"


def _legacy_surveillance_root(epiml_root: Path) -> Path:
    project_root = epiml_root / "project" / "surveillance"
    if project_root.exists():
        return project_root
    direct_root = epiml_root / "surveillance"
    if direct_root.exists():
        return direct_root
    return project_root


def _legacy_script_paths(epiml_root: Path, *, globs: list[str] | None = None) -> list[Path]:
    patterns = globs or ["*.R"]
    candidates = [
        _legacy_surveillance_root(epiml_root),
        epiml_root,
    ]
    scripts: list[Path] = []
    seen: set[str] = set()
    for root in candidates:
        if not root.exists():
            continue
        for pattern in patterns:
            for path in sorted(root.rglob(pattern)):
                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                scripts.append(path)
    return scripts


def _legacy_namespace_path(epiml_root: Path) -> Path:
    candidates = [
        epiml_root / "project" / "packages" / "epiml" / "NAMESPACE",
        epiml_root / "packages" / "epiml" / "NAMESPACE",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


@dataclass(frozen=True)
class ParitySummary:
    total_rows: int
    already_present: int
    scaffolded_incomplete: int
    missing: int
    deferred: int


def build_parity_matrix(
    epiml_root: str | Path,
    repo_root: str | Path | None = None,
    *,
    script_map: dict[str, dict[str, str]] | None = None,
    surface_map: dict[str, str] | None = None,
    file_globs: list[str] | None = None,
) -> pd.DataFrame:
    """Build a migration matrix comparing old and new codebases.

    When *script_map* and *surface_map* are ``None``, uses the
    built-in MORIE mappings.  Pass custom dicts to compare any
    two project trees.

    *file_globs* overrides the default ``["*.R"]`` pattern for
    discovering legacy scripts.
    """
    epiml_root = Path(epiml_root).expanduser().resolve()
    repo_root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()

    if not epiml_root.exists():
        raise FileNotFoundError(f"Legacy root not found: {epiml_root}")

    used_script_map = script_map if script_map is not None else MORIE_SCRIPT_MAP
    used_surface_map = surface_map if surface_map is not None else MORIE_SURFACE_MAP
    globs = file_globs or ["*.R"]

    rows: list[dict[str, str]] = []
    r_exports = _current_r_exports(repo_root)
    current_outputs = _current_output_files(repo_root)

    script_paths = _legacy_script_paths(epiml_root, globs=globs)
    for path in script_paths:
        name = path.name
        if name not in used_script_map:
            continue
        spec = used_script_map[name]
        status = spec["parity_status"]
        rows.append(
            {
                "kind": "analysis_module",
                "old_path": str(path.relative_to(epiml_root)),
                "required_inputs": "local CPADS wrangled data + module-specific covariates",
                "estimand_or_output": name.replace(".R", ""),
                "current_morie_status": status,
                "target_morie_api": spec["target_morie_api"],
                "target_docs_page": spec["target_docs_page"],
                "parity_status": status,
            }
        )

    old_namespace = _legacy_namespace_path(epiml_root)
    if old_namespace.exists():
        for line in old_namespace.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line.startswith("export("):
                continue
            export_name = line[len("export(") : -1]
            present = "morie_paths" in r_exports if export_name == "epiml_paths" else export_name in r_exports
            status = "already present" if present else "missing and must be migrated"
            rows.append(
                {
                    "kind": "r_package_utility",
                    "old_path": f"{old_namespace.parent.parent.relative_to(epiml_root)}/R/{export_name}",
                    "required_inputs": "package-level helper inputs",
                    "estimand_or_output": export_name,
                    "current_morie_status": used_surface_map.get(export_name, status),
                    "target_morie_api": export_name.replace("epiml_", "morie_"),
                    "target_docs_page": "api/index",
                    "parity_status": status,
                }
            )

    output_files: list[Path] = []
    for candidate in _output_dir_candidates(epiml_root):
        if candidate.exists():
            output_files.extend(candidate.rglob("*"))
    for path in _iter_unique_files([p for p in output_files if p.is_file() and not p.name.startswith(".")]):
        base_candidates = [
            candidate
            for candidate in _output_dir_candidates(epiml_root)
            if candidate.exists() and str(path).startswith(str(candidate))
        ]
        relative_name = str(path.relative_to(base_candidates[0])).replace("\\", "/") if base_candidates else path.name
        status = _published_output_status(relative_name, path, current_outputs)
        rows.append(
            {
                "kind": "published_output",
                "old_path": str(path.relative_to(epiml_root)),
                "required_inputs": "module-specific local CPADS analysis outputs",
                "estimand_or_output": relative_name,
                "current_morie_status": status,
                "target_morie_api": "module-compatible output writer",
                "target_docs_page": "migration",
                "parity_status": status,
            }
        )

    matrix = pd.DataFrame.from_records(rows).sort_values(["kind", "old_path"]).reset_index(drop=True)
    return matrix


def summarize_parity_matrix(matrix: pd.DataFrame) -> ParitySummary:
    """Summarize the migration-matrix status counts."""
    counts = matrix["parity_status"].value_counts()
    return ParitySummary(
        total_rows=len(matrix),
        already_present=int(counts.get("already present", 0)),
        scaffolded_incomplete=int(counts.get("scaffolded but statistically incomplete", 0)),
        missing=int(counts.get("missing and must be migrated", 0)),
        deferred=int(counts.get("intentionally deferred", 0)),
    )


def write_parity_matrix(matrix: pd.DataFrame, output_path: str | Path) -> Path:
    """Write the parity matrix to CSV."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_path, index=False)
    return output_path


def load_parity_config(config_path: str | Path) -> tuple[dict, dict]:
    """Load custom script_map and surface_map from a JSON config.

    Expected format::

        {
          "script_map": {"old.R": {"target_morie_api": "...", ...}},
          "surface_map": {"func_name": "status_string"}
        }
    """
    data = json.loads(Path(config_path).expanduser().resolve().read_text(encoding="utf-8"))
    return data.get("script_map", {}), data.get("surface_map", {})


def _get_parity_matrix() -> list[dict[str, Any]]:
    try:
        repo = Path(__file__).resolve().parents[2]
        legacy = repo / "migration_files" / "one"
        if not legacy.exists():
            return []
        df = build_parity_matrix(legacy, repo)
        return df.to_dict(orient="records")
    except (FileNotFoundError, Exception):
        return []


PARITY_MATRIX = _get_parity_matrix()


TIDE_PARITY = [
    {"feature": "Home Screen", "python_tui": "HomeScreen", "go_tide": "HomeModel", "status": "done"},
    {"feature": "Chat + Agent", "python_tui": "ChatScreen", "go_tide": "ChatModel", "status": "done"},
    {"feature": "REPL / IDE", "python_tui": "ReplScreen", "go_tide": "IDEModel", "status": "done"},
    {"feature": "Pipeline", "python_tui": "PipelineScreen", "go_tide": "PipelineModel", "status": "done"},
    {"feature": "Statistics Console", "python_tui": "StatScreen", "go_tide": "StatsModel", "status": "done"},
    {"feature": "Dataset Inspector", "python_tui": "DatasetScreen", "go_tide": "DatasetModel", "status": "done"},
    {"feature": "Help Topics", "python_tui": "HelpScreen", "go_tide": "HelpModel", "status": "done"},
    {"feature": "Doctor", "python_tui": "DoctorScreen", "go_tide": "DoctorModel", "status": "done"},
    {"feature": "Settings", "python_tui": "SettingsScreen", "go_tide": "SettingsModel", "status": "done"},
    {"feature": "Migration & Parity", "python_tui": "n/a", "go_tide": "MigrationModel", "status": "done"},
    {"feature": "Debug Screen", "python_tui": "DebugScreen", "go_tide": "merged into Doctor", "status": "done"},
    {"feature": "LLM Streaming", "python_tui": "paragraph flush", "go_tide": "StreamAsk channel", "status": "done"},
    {
        "feature": "Agent Mode (code exec)",
        "python_tui": "n/a",
        "go_tide": "agentMode + extractCodeBlock",
        "status": "done",
    },
    {
        "feature": "Polyglot Variable Bridge",
        "python_tui": "/polyglot P<->R<->Shell",
        "go_tide": "VarBridge + /polyglot",
        "status": "done",
    },
    {
        "feature": "1130+ fn/ Commands",
        "python_tui": "stat_commands.py",
        "go_tide": "stat_bridge exec",
        "status": "done",
    },
    {
        "feature": "Slash Commands (35+)",
        "python_tui": "14 base",
        "go_tide": "35+ via HandleCLICommand",
        "status": "done",
    },
    {"feature": "Clipboard/Export", "python_tui": "F2/ctrl+o", "go_tide": "/copy + /export + OSC52", "status": "done"},
    {"feature": "File Tree Browser", "python_tui": "n/a", "go_tide": "TreeModel in Doctor+REPL", "status": "done"},
    {
        "feature": "10+ Language Support",
        "python_tui": "Python+R+Shell",
        "go_tide": "Python+R+Shell+Go+TS+JS+HTML+CSS+LaTeX+Lua",
        "status": "done",
    },
    {
        "feature": "Causal Inference",
        "python_tui": "8 methods",
        "go_tide": "9 methods (+ synth control)",
        "status": "done",
    },
    {"feature": "Dataset Context → Chat", "python_tui": "loaded_df injection", "go_tide": "n/a", "status": "missing"},
    {"feature": "Agent Personas", "python_tui": "/agent <name>", "go_tide": "single agent only", "status": "partial"},
]


def build_tide_parity() -> pd.DataFrame:
    """Build parity matrix comparing Python TUI screens to Go TIDE screens."""
    return pd.DataFrame(TIDE_PARITY)


def summarize_tide_parity() -> dict[str, int]:
    """Count done/partial/missing features in TIDE parity."""
    counts: dict[str, int] = {"done": 0, "partial": 0, "missing": 0}
    for row in TIDE_PARITY:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return counts
