"""Sphinx documentation helpers for package-facing MORIE pages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .modules import MODULE_SPECS


@dataclass(frozen=True)
class ModuleDocSpec:
    title: str
    group: str
    script_path: str
    python_helper: str | None = None


MODULE_DOC_SPECS: dict[str, ModuleDocSpec] = {
    "data-wrangling": ModuleDocSpec(
        "Data Wrangling", "investigation", "surveillance/investigation/03_data_wrangling.R"
    ),
    "descriptive-statistics": ModuleDocSpec(
        "Descriptive Statistics", "investigation", "surveillance/investigation/04_descriptive_stats.R"
    ),
    "distribution-tests": ModuleDocSpec(
        "Distribution Tests", "investigation", "surveillance/investigation/04_distributions.R"
    ),
    "frequentist-inference": ModuleDocSpec(
        "Frequentist Inference", "investigation", "surveillance/investigation/05_frequentist.R"
    ),
    "bayesian-inference": ModuleDocSpec(
        "Bayesian Inference", "investigation", "surveillance/investigation/05_bayesian.R"
    ),
    "power-design": ModuleDocSpec(
        "Power Design", "investigation", "surveillance/investigation/05_power_design.R", "morie.run_power_design_module"
    ),
    "logistic-models": ModuleDocSpec(
        "Logistic Models",
        "investigation",
        "surveillance/investigation/06_logistic.R",
        "morie.run_weighted_logistic_analysis",
    ),
    "model-comparison": ModuleDocSpec(
        "Model Comparison",
        "investigation",
        "surveillance/investigation/06_model_comparison.R",
        "morie.compare_nested_logistic_models",
    ),
    "regression-models": ModuleDocSpec(
        "Regression Models", "investigation", "surveillance/investigation/06_regression.R"
    ),
    "propensity-scores": ModuleDocSpec(
        "Propensity Scores",
        "investigation",
        "surveillance/investigation/07_propensity.R",
        "morie.run_propensity_ipw_analysis",
    ),
    "causal-estimators": ModuleDocSpec(
        "Causal Estimators", "investigation", "surveillance/investigation/07_causal_estimators.R"
    ),
    "treatment-effects": ModuleDocSpec(
        "Treatment Effects",
        "investigation",
        "surveillance/investigation/07_treatment_effects.R",
        "morie.run_treatment_effects_analysis",
    ),
    "dag-specification": ModuleDocSpec("DAG Specification", "investigation", "surveillance/investigation/07_dag.R"),
    "meta-synthesis": ModuleDocSpec(
        "Meta Synthesis", "investigation", "surveillance/investigation/07_meta_synthesis.R"
    ),
    "figures": ModuleDocSpec("Figures", "investigation", "surveillance/investigation/08_figures.R"),
    "tables": ModuleDocSpec("Tables", "investigation", "surveillance/investigation/08_tables.R"),
    "final-report": ModuleDocSpec("Final Report", "investigation", "surveillance/investigation/09_report.R"),
    "ebac-core": ModuleDocSpec("eBAC Core", "ebac", "surveillance/ebac/07_ebac.R"),
    "ebac-selection-adjustment-ipw": ModuleDocSpec(
        "eBAC Selection Adjustment (IPW)",
        "ebac",
        "surveillance/ebac/07_ebac_ipw.R",
        "morie.run_ebac_selection_ipw_analysis",
    ),
    "ebac-integrations": ModuleDocSpec("eBAC Integrations", "ebac", "surveillance/ebac/07_ebac_integrations.R"),
    "ebac-gender-smote-sensitivity": ModuleDocSpec(
        "eBAC Gender and SMOTE Sensitivity", "ebac", "surveillance/ebac/07_ebac_gender_smote_sensitivity.R"
    ),
}

GROUP_ORDER = ("investigation", "ebac")

GROUP_TITLES = {
    "investigation": "Investigation Modules",
    "ebac": "eBAC Modules",
}

GROUP_INTROS = {
    "investigation": (
        "These modules cover CPADS wrangling, inferential analysis, model comparison, "
        "causal estimation, reporting, and supporting figures/tables."
    ),
    "ebac": (
        "These modules cover eBAC-specific workflow steps, including weighted summaries, "
        "selection adjustment, interaction checks, and integration outputs."
    ),
}

MODULE_DOWNLOAD_NOTES = {
    "power-design": [
        ("Power example page", "../auto_examples/plot_power_design.html"),
        ("Python example source", "../_downloads/b1b35979ca00b8a8b724457f8402d14c/plot_power_design.py"),
        ("Jupyter notebook", "../_downloads/933e4ae199998932c3d86eba7f72bb3f/plot_power_design.ipynb"),
    ]
}


def _repo_root(repo_root: str | Path | None = None) -> Path:
    if repo_root is not None:
        return Path(repo_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _read_manifest(repo_root: Path) -> pd.DataFrame:
    manifest_path = repo_root / "data" / "public" / "outputs_manifest.csv"
    if not manifest_path.exists():
        return pd.DataFrame(columns=["output", "public_path", "size_kb", "modified"])
    return pd.read_csv(manifest_path)


def _module_rows(group: str, manifest: pd.DataFrame) -> list[str]:
    lines: list[str] = [
        f"# {GROUP_TITLES[group]}",
        "",
        GROUP_INTROS[group],
        "",
    ]

    specs = [
        (module_name, MODULE_DOC_SPECS[module_name], MODULE_SPECS[module_name])
        for module_name in MODULE_SPECS
        if MODULE_DOC_SPECS[module_name].group == group
    ]

    for module_name, doc_spec, module_spec in specs:
        lines.extend(
            [
                f"## {doc_spec.title}",
                "",
                module_spec.description,
                "",
                "**Entrypoints**",
                "",
                f"- CLI: `morie run-module {module_name} --cpads-csv <path> --output-dir <dir>`",
                f'- Python: `morie.run_module("{module_name}", cpads_csv=..., output_dir=...)`',
                f'- R: `run_morie_module("{module_name}", cpads_csv = ..., output_dir = ...)`',
            ]
        )
        if doc_spec.python_helper:
            lines.append(f"- Direct Python helper: `{doc_spec.python_helper}(...)`")
        lines.extend(
            [
                f"- Workflow script: `{doc_spec.script_path}`",
                "",
                "**Expected outputs**",
                "",
            ]
        )

        if module_spec.output_files:
            for output_name in module_spec.output_files:
                lines.append(f"- `{output_name}`")
        else:
            lines.append("- This module publishes workflow side effects rather than standalone public artifacts.")

        lines.extend(
            [
                "",
                f"Artifact inventory: [current published files](../project/artifacts.html#{module_name})",
                "",
            ]
        )

        if module_name in MODULE_DOWNLOAD_NOTES:
            lines.extend(["**Downloads**", ""])
            for label, href in MODULE_DOWNLOAD_NOTES[module_name]:
                lines.append(f"- [{label}]({href})")
            lines.append("")

        published = manifest.loc[manifest["output"].isin(module_spec.output_files), "output"].tolist()
        if published:
            lines.extend(["**Currently published**", ""])
            for output_name in published:
                lines.append(f"- `{output_name}`")
            lines.append("")

    return lines


def _build_artifact_rows(module_name: str, manifest: pd.DataFrame) -> list[str]:
    module_spec = MODULE_SPECS[module_name]
    title = MODULE_DOC_SPECS[module_name].title
    expected = list(module_spec.output_files)
    lines = [f"## {title}", "", f'<a id="{module_name}"></a>', ""]

    if not expected:
        lines.extend(["This module does not declare standalone public artifacts.", ""])
        return lines

    lines.extend(
        [
            "| Output | Status | Open |",
            "| --- | --- | --- |",
        ]
    )
    present = {row.output: row for row in manifest.itertuples(index=False)}
    for output_name in expected:
        row = present.get(output_name)
        if row is None:
            lines.append(f"| `{output_name}` | Not currently published | -- |")
            continue
        href = f"../outputs/{output_name}"
        lines.append(f"| `{output_name}` | Published | [Open]({href}) |")
    lines.append("")
    return lines


def write_module_guide_pages(docs_root: str | Path, repo_root: str | Path | None = None) -> list[Path]:
    repo = _repo_root(repo_root)
    docs_root = Path(docs_root)
    docs_root.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(repo)
    written: list[Path] = []
    for group in GROUP_ORDER:
        page_path = docs_root / f"{group}.md"
        page_path.write_text("\n".join(_module_rows(group, manifest)) + "\n", encoding="utf-8")
        written.append(page_path)
    return written


def write_artifact_inventory_page(docs_path: str | Path, repo_root: str | Path | None = None) -> Path:
    repo = _repo_root(repo_root)
    docs_path = Path(docs_path)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(repo)
    lines = [
        "# Artifacts And Output Contracts",
        "",
        "This page lists the currently published output files for each MORIE module.",
        "It is an artifact inventory for package users and maintainers, not a narrative results report.",
        "",
        "- Manifest file: [outputs_manifest.csv](../outputs_manifest.csv)",
        "- Public artifact root: `data/manifest/outputs/`",
        "",
    ]

    for group in GROUP_ORDER:
        lines.append(f"## {GROUP_TITLES[group]}")
        lines.append("")
        for module_name in MODULE_SPECS:
            if MODULE_DOC_SPECS[module_name].group != group:
                continue
            lines.extend(_build_artifact_rows(module_name, manifest))

    docs_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return docs_path
