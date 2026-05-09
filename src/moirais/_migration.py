"""General-purpose migration inventory and directory comparison engine.

Two modes:
  1. General: ``compare_trees(source, target)`` diffs any two directory trees.
  2. MOIRAIS default: ``build_migration_inventory()`` uses built-in MOIRAIS mappings.

Example (general)::

    from moirais._migration import compare_trees, scan_directory
    inv = scan_directory("/path/to/legacy")
    df = compare_trees("/path/to/legacy", "/path/to/modern")

Example (MOIRAIS-specific)::

    from moirais._migration import build_migration_inventory
    df = build_migration_inventory()
"""

from __future__ import annotations

import ast
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

CODE_EXTENSIONS = {
    ".py",
    ".r",
    ".R",
    ".go",
    ".js",
    ".ts",
    ".c",
    ".cpp",
    ".h",
    ".rs",
    ".java",
    ".rb",
    ".jl",
    ".lua",
    ".sh",
    ".bash",
    ".zsh",
    ".qmd",
    ".Rmd",
    ".sql",
}


def scan_directory(
    root: str | Path,
    *,
    extensions: set[str] | None = None,
    exclude_dirs: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Inventory all code files in a directory tree.

    :param root: Root directory to scan.
    :param extensions: File extensions to include (default: common code extensions).
    :param exclude_dirs: Directory names to skip (default: .git, __pycache__, node_modules, .venv).
    :return: ``{relative_path: {ext, size, lines, functions}}``
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")

    exts = extensions or CODE_EXTENSIONS
    skip = exclude_dirs or {".git", "__pycache__", "node_modules", ".venv", "venv", ".tox", ".eggs"}

    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in skip for part in path.parts):
            continue
        if path.suffix not in exts:
            continue

        rel = str(path.relative_to(root))
        size = path.stat().st_size
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            lines = text.count("\n") + 1
            functions = _extract_function_names(path, text)
        except Exception:
            lines = 0
            functions = []

        inventory[rel] = {
            "ext": path.suffix,
            "size": size,
            "lines": lines,
            "functions": functions,
        }

    return inventory


def _extract_function_names(path: Path, text: str) -> list[str]:
    """Extract function/method names from a source file."""
    suffix = path.suffix.lower()
    if suffix == ".py":
        return _extract_python_functions(text)
    if suffix in {".r", ".rmd", ".qmd"}:
        return _extract_r_functions(text)
    if suffix == ".go":
        return _extract_go_functions(text)
    if suffix in {".js", ".ts"}:
        return _extract_js_functions(text)
    return []


def _extract_python_functions(text: str) -> list[str]:
    try:
        tree = ast.parse(text)
        return [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_")
        ]
    except SyntaxError:
        return re.findall(r"^def\s+(\w+)", text, re.MULTILINE)


def _extract_r_functions(text: str) -> list[str]:
    return re.findall(r"^(\w+)\s*<-\s*function\s*\(", text, re.MULTILINE)


def _extract_go_functions(text: str) -> list[str]:
    return re.findall(r"^func\s+(?:\([^)]*\)\s+)?(\w+)\s*\(", text, re.MULTILINE)


def _extract_js_functions(text: str) -> list[str]:
    named = re.findall(r"^(?:export\s+)?function\s+(\w+)\s*\(", text, re.MULTILINE)
    arrows = re.findall(r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(", text, re.MULTILINE)
    return named + arrows


def auto_match(
    source: dict[str, dict[str, Any]],
    target: dict[str, dict[str, Any]],
    *,
    threshold: float = 0.5,
) -> list[tuple[str, str, float]]:
    """Fuzzy-match source files to target files by name and function overlap.

    :return: List of ``(source_path, target_path, score)`` tuples.
    """
    matches = []
    used_targets: set[str] = set()

    for s_path, s_info in source.items():
        s_stem = Path(s_path).stem.lower()
        s_fns = set(s_info.get("functions", []))

        best_score = 0.0
        best_target = ""

        for t_path, t_info in target.items():
            if t_path in used_targets:
                continue
            t_stem = Path(t_path).stem.lower()
            t_fns = set(t_info.get("functions", []))

            name_sim = SequenceMatcher(None, s_stem, t_stem).ratio()
            fn_overlap = len(s_fns & t_fns) / max(len(s_fns | t_fns), 1) if s_fns and t_fns else 0.0

            score = 0.6 * name_sim + 0.4 * fn_overlap
            if score > best_score:
                best_score = score
                best_target = t_path

        if best_score >= threshold and best_target:
            matches.append((s_path, best_target, best_score))
            used_targets.add(best_target)

    return sorted(matches, key=lambda x: -x[2])


def compare_trees(
    source_dir: str | Path,
    target_dir: str | Path,
    *,
    mapping: dict[str, dict[str, str]] | None = None,
    extensions: set[str] | None = None,
    auto_match_threshold: float = 0.5,
) -> pd.DataFrame:
    """Diff two directory trees and produce a migration matrix.

    :param source_dir: Legacy / source directory.
    :param target_dir: Modern / target directory.
    :param mapping: Optional ``{source_file: {target, status, notes}}`` overrides.
    :param extensions: File extensions to scan.
    :param auto_match_threshold: Minimum similarity for auto-matching (0-1).
    :return: DataFrame with columns: source_file, target_file, status,
        source_functions, target_functions, match_score.
    """
    source = scan_directory(source_dir, extensions=extensions)
    target = scan_directory(target_dir, extensions=extensions)

    rows: list[dict[str, Any]] = []
    matched_sources: set[str] = set()
    matched_targets: set[str] = set()

    if mapping:
        for s_file, spec in mapping.items():
            t_file = spec.get("target", "")
            status = spec.get("status", "mapped")
            s_info = source.get(s_file, {})
            t_info = target.get(t_file, {}) if t_file else {}
            rows.append(
                {
                    "source_file": s_file,
                    "target_file": t_file,
                    "status": status,
                    "source_functions": len(s_info.get("functions", [])),
                    "target_functions": len(t_info.get("functions", [])),
                    "match_score": 1.0 if t_file else 0.0,
                    "notes": spec.get("notes", ""),
                }
            )
            matched_sources.add(s_file)
            if t_file:
                matched_targets.add(t_file)

    unmatched_source = {k: v for k, v in source.items() if k not in matched_sources}
    unmatched_target = {k: v for k, v in target.items() if k not in matched_targets}
    auto_matches = auto_match(unmatched_source, unmatched_target, threshold=auto_match_threshold)

    for s_file, t_file, score in auto_matches:
        s_fns = set(source[s_file].get("functions", []))
        t_fns = set(target[t_file].get("functions", []))
        overlap = len(s_fns & t_fns)
        if s_fns and t_fns and overlap == len(s_fns):
            status = "full_parity"
        elif overlap > 0:
            status = "partial_parity"
        else:
            status = "name_match_only"

        rows.append(
            {
                "source_file": s_file,
                "target_file": t_file,
                "status": status,
                "source_functions": len(s_fns),
                "target_functions": len(t_fns),
                "match_score": round(score, 3),
                "notes": "",
            }
        )
        matched_sources.add(s_file)
        matched_targets.add(t_file)

    for s_file in sorted(set(source) - matched_sources):
        s_info = source[s_file]
        rows.append(
            {
                "source_file": s_file,
                "target_file": "",
                "status": "source_only",
                "source_functions": len(s_info.get("functions", [])),
                "target_functions": 0,
                "match_score": 0.0,
                "notes": "",
            }
        )

    for t_file in sorted(set(target) - matched_targets):
        t_info = target[t_file]
        rows.append(
            {
                "source_file": "",
                "target_file": t_file,
                "status": "target_only",
                "source_functions": 0,
                "target_functions": len(t_info.get("functions", [])),
                "match_score": 0.0,
                "notes": "",
            }
        )

    return pd.DataFrame(rows).sort_values("source_file").reset_index(drop=True)


def load_mapping_from_json(config_path: str | Path) -> dict[str, dict[str, str]]:
    """Load a migration mapping from a JSON configuration file."""
    data = json.loads(Path(config_path).expanduser().resolve().read_text(encoding="utf-8"))
    return data


# ── MOIRAIS-specific defaults ──────────────────────────────────────────────────

LEGACY_ROOT = Path("migration_files/one")

MOIRAIS_MODULE_DOCS = {
    "03_data_wrangling.R": "modules/investigation",
    "04_descriptive_stats.R": "modules/investigation",
    "04_distributions.R": "modules/investigation",
    "05_frequentist.R": "modules/investigation",
    "05_bayesian.R": "modules/investigation",
    "05_power_design.R": "modules/investigation",
    "06_logistic.R": "modules/investigation",
    "06_model_comparison.R": "modules/investigation",
    "06_regression.R": "modules/investigation",
    "07_propensity.R": "modules/investigation",
    "07_causal_estimators.R": "modules/investigation",
    "07_treatment_effects.R": "modules/investigation",
    "07_dag.R": "modules/investigation",
    "07_meta_synthesis.R": "modules/investigation",
    "08_figures.R": "modules/investigation",
    "08_tables.R": "modules/investigation",
    "09_report.R": "modules/investigation",
    "07_ebac.R": "modules/ebac",
    "07_ebac_ipw.R": "modules/ebac",
    "07_ebac_gender_smote_sensitivity.R": "modules/ebac",
    "07_ebac_integrations.R": "modules/ebac",
}

MOIRAIS_LEGACY_TO_MODULE = {
    "03_data_wrangling.R": "data-wrangling",
    "04_descriptive_stats.R": "descriptive-statistics",
    "04_distributions.R": "distribution-tests",
    "05_frequentist.R": "frequentist-inference",
    "05_bayesian.R": "bayesian-inference",
    "05_power_design.R": "power-design",
    "06_logistic.R": "logistic-models",
    "06_model_comparison.R": "model-comparison",
    "06_regression.R": "regression-models",
    "07_propensity.R": "propensity-scores",
    "07_causal_estimators.R": "causal-estimators",
    "07_treatment_effects.R": "treatment-effects",
    "07_dag.R": "dag-specification",
    "07_meta_synthesis.R": "meta-synthesis",
    "07_ebac.R": "ebac-core",
    "07_ebac_ipw.R": "ebac-selection-adjustment-ipw",
    "07_ebac_gender_smote_sensitivity.R": "ebac-gender-smote-sensitivity",
    "07_ebac_integrations.R": "ebac-integrations",
    "08_figures.R": "figures",
    "08_tables.R": "tables",
    "09_report.R": "final-report",
}

MOIRAIS_IMPLEMENTATION_OWNER = {v: "R" for v in MOIRAIS_LEGACY_TO_MODULE.values()}

LEGACY_MODULE_DOCS = MOIRAIS_MODULE_DOCS
LEGACY_TO_MODULE = MOIRAIS_LEGACY_TO_MODULE
IMPLEMENTATION_OWNER = MOIRAIS_IMPLEMENTATION_OWNER


def build_migration_inventory(
    repo_root: str | Path | None = None,
    legacy_root: str | Path | None = None,
    *,
    mapping: dict[str, dict[str, str]] | None = None,
) -> pd.DataFrame:
    """Build a migration inventory (MOIRAIS default or custom mapping)."""
    if mapping is not None:
        source = Path(legacy_root) if legacy_root else Path(".")
        target_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
        return compare_trees(source, target_root, mapping=mapping)

    repo = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    legacy = Path(legacy_root) if legacy_root else repo / LEGACY_ROOT

    rows: list[dict[str, Any]] = []
    for script, module in sorted(MOIRAIS_LEGACY_TO_MODULE.items()):
        rows.append(
            {
                "legacy_script": script,
                "module_name": module,
                "implementation_owner": MOIRAIS_IMPLEMENTATION_OWNER.get(module, "unknown"),
                "docs_page": MOIRAIS_MODULE_DOCS.get(script, ""),
                "legacy_exists": (legacy / script).exists() if legacy.exists() else False,
            }
        )

    return pd.DataFrame(rows).sort_values("module_name").reset_index(drop=True)


def _get_migration_inventory() -> list[dict[str, Any]]:
    try:
        df = build_migration_inventory()
        return df.to_dict(orient="records")
    except Exception:
        return []


MIGRATION_INVENTORY = _get_migration_inventory()
