"""Notebook management for MOIRAIS.

Provides utilities for creating, rendering, converting, validating, and
managing Quarto (``.qmd``), Jupyter (``.ipynb``), and R Markdown (``.Rmd``)
notebooks.  Functions wrap the ``quarto`` and ``jupyter`` CLIs with progress
display, error handling, and structured output.

The module supports:

* Template-based notebook creation for common epidemiological analyses.
* Parameterized notebook execution (pass variables at render time).
* Batch rendering with progress tracking.
* Format conversion between ``.qmd``, ``.ipynb``, and ``.Rmd``.
* Notebook validation (check that all cells execute without error).
* Auto-generation of analysis notebooks from pipeline results.

References
----------
Allaire, J. J. et al. (2022). *Quarto: An Open-Source Scientific and
Technical Publishing System*. https://quarto.org

Kluyver, T. et al. (2016). Jupyter Notebooks -- a publishing format for
reproducible computational workflows. *Positioning and Power in Academic
Publishing*, 87--90. https://doi.org/10.3233/978-1-61499-649-1-87
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_FORMATS = {".qmd", ".ipynb", ".Rmd", ".rmd"}
RENDER_FORMATS = {"html", "pdf", "docx", "revealjs", "pptx", "gfm", "epub"}

_NOTEBOOK_DIR_ENVVAR = "MOIRAIS_NOTEBOOK_DIR"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class NotebookInfo:
    """Metadata for a notebook file."""

    path: str
    filename: str
    format: str
    title: str | None
    size_bytes: int
    modified: str
    cell_count: int = 0
    code_cell_count: int = 0
    has_errors: bool = False


@dataclass
class RenderResult:
    """Result from rendering a notebook."""

    source: str
    output_path: str
    output_format: str
    success: bool
    duration_seconds: float
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0


@dataclass
class ConversionResult:
    """Result from a format conversion."""

    source: str
    target: str
    source_format: str
    target_format: str
    success: bool
    message: str = ""


@dataclass
class ValidationResult:
    """Result from validating a notebook."""

    path: str
    valid: bool
    total_cells: int
    executed_cells: int
    failed_cells: int
    errors: list[dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0


@dataclass
class NotebookCatalog:
    """Catalog of notebooks in the project."""

    notebooks: list[NotebookInfo]
    total_count: int
    by_format: dict[str, int]
    scan_directory: str


# ---------------------------------------------------------------------------
# CLI detection
# ---------------------------------------------------------------------------


def _quarto_available() -> bool:
    """Check if Quarto CLI is available."""
    return shutil.which("quarto") is not None


def _jupyter_available() -> bool:
    """Check if Jupyter/nbconvert is available."""
    return shutil.which("jupyter") is not None


def _run_command(
    cmd: list[str],
    *,
    timeout: int = 600,
    cwd: str | Path | None = None,
) -> tuple[int, str, str, float]:
    """Run a subprocess and return (returncode, stdout, stderr, duration)."""
    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
        )
        elapsed = time.monotonic() - start
        return result.returncode, result.stdout, result.stderr, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        return -1, "", f"Command timed out after {timeout}s", elapsed
    except FileNotFoundError:
        elapsed = time.monotonic() - start
        return -1, "", f"Command not found: {cmd[0]}", elapsed


# ---------------------------------------------------------------------------
# Notebook discovery
# ---------------------------------------------------------------------------


def list_notebooks(
    directory: str | Path = ".",
    *,
    recursive: bool = True,
    formats: set[str] | None = None,
) -> list[NotebookInfo]:
    """List all notebooks in a directory.

    Parameters
    ----------
    directory : str | Path
        Directory to scan (default: current directory).
    recursive : bool
        If True, search subdirectories.
    formats : set[str] | None
        File extensions to include (default: all supported formats).

    Returns
    -------
    list[NotebookInfo]
        Sorted by path.
    """
    directory = Path(directory).resolve()
    allowed = formats or SUPPORTED_FORMATS
    notebooks: list[NotebookInfo] = []

    pattern = "**/*" if recursive else "*"
    for p in sorted(directory.glob(pattern)):
        if p.suffix.lower() not in allowed or not p.is_file():
            continue

        title = _extract_title(p)
        cell_count = 0
        code_cells = 0

        if p.suffix.lower() == ".ipynb":
            try:
                nb = json.loads(p.read_text(encoding="utf-8"))
                cells = nb.get("cells", [])
                cell_count = len(cells)
                code_cells = sum(1 for c in cells if c.get("cell_type") == "code")
            except (json.JSONDecodeError, OSError):
                pass
        elif p.suffix.lower() == ".qmd":
            content = p.read_text(encoding="utf-8", errors="replace")
            # Count code chunks (```{python}, ```{r})
            lines = content.splitlines()
            in_chunk = False
            for line in lines:
                if line.strip().startswith("```{"):
                    in_chunk = True
                    code_cells += 1
                    cell_count += 1
                elif line.strip() == "```" and in_chunk:
                    in_chunk = False
                elif not in_chunk and line.strip():
                    pass  # markdown content

        stat = p.stat()
        notebooks.append(
            NotebookInfo(
                path=str(p),
                filename=p.name,
                format=p.suffix.lower(),
                title=title,
                size_bytes=stat.st_size,
                modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                cell_count=cell_count,
                code_cell_count=code_cells,
            )
        )

    return notebooks


def build_catalog(
    directory: str | Path = ".",
    *,
    recursive: bool = True,
) -> NotebookCatalog:
    """Build a catalog of all notebooks in a directory.

    Parameters
    ----------
    directory : str | Path
        Directory to scan.
    recursive : bool
        Search subdirectories.

    Returns
    -------
    NotebookCatalog
    """
    nbs = list_notebooks(directory, recursive=recursive)
    by_format: dict[str, int] = {}
    for nb in nbs:
        by_format[nb.format] = by_format.get(nb.format, 0) + 1

    return NotebookCatalog(
        notebooks=nbs,
        total_count=len(nbs),
        by_format=by_format,
        scan_directory=str(Path(directory).resolve()),
    )


def _extract_title(path: Path) -> str | None:
    """Extract title from a notebook's YAML frontmatter or first heading."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    if path.suffix.lower() == ".ipynb":
        try:
            nb = json.loads(content)
            for cell in nb.get("cells", []):
                if cell.get("cell_type") == "markdown":
                    source = "".join(cell.get("source", []))
                    for line in source.splitlines():
                        if line.startswith("# "):
                            return line[2:].strip()
        except json.JSONDecodeError:
            pass
        return None

    # QMD / RMD: check YAML frontmatter
    lines = content.splitlines()
    in_yaml = False
    for line in lines:
        stripped = line.strip()
        if stripped == "---":
            if not in_yaml:
                in_yaml = True
                continue
            else:
                break
        if in_yaml and stripped.startswith("title:"):
            title = stripped[6:].strip().strip("'\"")
            return title if title else None

    # Fallback: first markdown heading
    for line in lines:
        if line.startswith("# "):
            return line[2:].strip()

    return None


# ---------------------------------------------------------------------------
# Template-based notebook creation
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, str] = {}

_TEMPLATES["analysis"] = textwrap.dedent("""\
    ---
    title: "{title}"
    author: "The art of doing mathematics consists in finding that special case which contains all the germs of generality. — David Hilbert"
    date: "{date}"
    format:
      html:
        toc: true
        code-fold: true
        embed-resources: true
    execute:
      warning: false
      message: false
    ---

    ## Overview

    This notebook presents a statistical analysis using the MOIRAIS framework.

    ## Setup

    ```{{python}}
    import pandas as pd
    import numpy as np
    import moirais
    ```

    ## Data Loading

    ```{{python}}
    # Load your dataset
    # df = pd.read_csv("data/your_data.csv")
    # profile = moirais.profile_dataset(df)
    ```

    ## Descriptive Statistics

    ```{{python}}
    # df.describe()
    ```

    ## Analysis

    ```{{python}}
    # Your analysis code here
    ```

    ## Results

    Summarize findings here.

    ## Discussion

    Interpret results in context.
""")

_TEMPLATES["report"] = textwrap.dedent("""\
    ---
    title: "{title}"
    author: "The art of doing mathematics consists in finding that special case which contains all the germs of generality. — David Hilbert"
    date: "{date}"
    format:
      html:
        toc: true
        toc-depth: 3
        embed-resources: true
        theme: cosmo
      pdf:
        documentclass: article
        geometry: margin=1in
    execute:
      echo: false
      warning: false
    ---

    ## Executive Summary

    Brief summary of findings.

    ## Introduction

    ### Background

    ### Objectives

    ## Methods

    ### Study Design

    ### Data Sources

    ### Statistical Analysis

    ```{{python}}
    #| echo: false
    import pandas as pd
    import numpy as np
    import moirais
    ```

    ## Results

    ### Sample Description

    ### Primary Analysis

    ### Sensitivity Analyses

    ## Discussion

    ### Key Findings

    ### Limitations

    ### Implications

    ## References
""")

_TEMPLATES["causal"] = textwrap.dedent("""\
    ---
    title: "{title}"
    author: "The art of doing mathematics consists in finding that special case which contains all the germs of generality. — David Hilbert"
    date: "{date}"
    format:
      html:
        toc: true
        code-fold: true
        embed-resources: true
    execute:
      warning: false
    ---

    ## Causal Analysis: {title}

    ### Setup

    ```{{python}}
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    import moirais
    from moirais.causal import compute_propensity_scores, calculate_ipw_weights
    from moirais.effects import estimate_ate
    ```

    ### DAG Specification

    Describe the assumed causal structure.

    ### Data Preparation

    ```{{python}}
    # df = pd.read_csv("data/your_data.csv")
    # treatment = "treatment_variable"
    # outcome = "outcome_variable"
    # covariates = ["X1", "X2", "X3"]
    ```

    ### Propensity Score Estimation

    ```{{python}}
    # ps = moirais.compute_propensity_scores(df, treatment, covariates)
    ```

    ### IPW Analysis

    ```{{python}}
    # weights = moirais.calculate_ipw_weights(ps)
    ```

    ### Double Machine Learning

    ```{{python}}
    # ate_result = moirais.estimate_ate(df, outcome, treatment, covariates)
    ```

    ### Sensitivity Analysis

    ### Results Summary
""")

_TEMPLATES["table1"] = textwrap.dedent("""\
    ---
    title: "Table 1: Baseline Characteristics — {title}"
    author: "The art of doing mathematics consists in finding that special case which contains all the germs of generality. — David Hilbert"
    date: "{date}"
    format:
      html:
        toc: false
        embed-resources: true
    execute:
      echo: false
      warning: false
    ---

    ```{{python}}
    import pandas as pd
    import numpy as np
    import moirais

    # df = pd.read_csv("data/your_data.csv")
    # group_var = "treatment"
    ```

    ```{{python}}
    # Continuous variables: mean (SD)
    # Categorical variables: n (%)
    # Include p-values from appropriate tests
    ```
""")

_TEMPLATES["survival"] = textwrap.dedent("""\
    ---
    title: "{title}"
    author: "The art of doing mathematics consists in finding that special case which contains all the germs of generality. — David Hilbert"
    date: "{date}"
    format:
      html:
        toc: true
        code-fold: true
        embed-resources: true
    ---

    ## Survival Analysis

    ```{{python}}
    import pandas as pd
    import numpy as np
    from scipy import stats
    import moirais
    ```

    ### Data Preparation

    ```{{python}}
    # Define time-to-event and censoring variables
    # time_var = "follow_up_days"
    # event_var = "event_occurred"
    ```

    ### Kaplan-Meier Estimates

    ### Log-Rank Test

    ### Cox Proportional Hazards

    ### Diagnostics

    ### Results
""")

_TEMPLATES["presentation"] = textwrap.dedent("""\
    ---
    title: "{title}"
    author: "The art of doing mathematics consists in finding that special case which contains all the germs of generality. — David Hilbert"
    date: "{date}"
    format:
      revealjs:
        theme: simple
        slide-number: true
        embed-resources: true
    ---

    ## Background

    - Point 1
    - Point 2

    ## Methods

    ```{{python}}
    #| echo: false
    import moirais
    ```

    ## Results

    ## Conclusions

    ## Questions?
""")


def create_notebook(
    path: str | Path,
    *,
    template: str = "analysis",
    title: str = "Untitled Analysis",
    author: str = "MOIRAIS",
    date: str | None = None,
    format: str = "qmd",
    overwrite: bool = False,
    custom_content: str | None = None,
) -> Path:
    """Create a new notebook from a template.

    Parameters
    ----------
    path : str | Path
        Output file path (extension added if missing).
    template : str
        Template name: ``analysis``, ``report``, ``causal``, ``table1``,
        ``survival``, ``presentation``.
    title : str
        Notebook title.
    author : str
        Author name.
    date : str | None
        Date string (default: today).
    format : str
        Output format: ``qmd``, ``ipynb``, ``Rmd``.
    overwrite : bool
        If True, overwrite existing file.
    custom_content : str | None
        If provided, use this as the notebook body instead of a template.

    Returns
    -------
    Path
        Path to the created notebook.

    Raises
    ------
    ValueError
        If the template name is unknown.
    FileExistsError
        If the file exists and ``overwrite`` is False.
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    path = Path(path)
    if path.suffix == "":
        path = path.with_suffix(f".{format}")

    if path.exists() and not overwrite:
        raise FileExistsError(f"Notebook already exists: {path}. Use overwrite=True.")

    if custom_content is not None:
        content = custom_content
    elif template in _TEMPLATES:
        content = _TEMPLATES[template].format(title=title, author=author, date=date)
    else:
        available = ", ".join(sorted(_TEMPLATES.keys()))
        raise ValueError(f"Unknown template '{template}'. Available: {available}")

    # If target format is ipynb, convert from qmd content
    if path.suffix.lower() == ".ipynb":
        nb = _qmd_to_ipynb_content(content, title=title)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    logger.info("Created notebook: %s (template=%s)", path, template)
    return path


def _qmd_to_ipynb_content(qmd_text: str, *, title: str = "") -> dict[str, Any]:
    """Convert QMD text content to an ipynb JSON structure."""
    cells: list[dict[str, Any]] = []
    lines = qmd_text.splitlines()
    current_chunk: list[str] = []
    current_type = "markdown"
    in_code = False
    in_yaml = False

    for line in lines:
        stripped = line.strip()
        if stripped == "---":
            if not in_yaml and not cells and not current_chunk:
                in_yaml = True
                continue
            elif in_yaml:
                in_yaml = False
                continue
        if in_yaml:
            continue

        if stripped.startswith("```{") and not in_code:
            # Flush markdown
            if current_chunk:
                cells.append(
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [l + "\n" for l in current_chunk],
                    }
                )
                current_chunk = []
            in_code = True
            continue
        elif stripped == "```" and in_code:
            # Flush code
            cells.append(
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [l + "\n" for l in current_chunk],
                    "outputs": [],
                    "execution_count": None,
                }
            )
            current_chunk = []
            in_code = False
            continue

        current_chunk.append(line)

    # Flush remaining
    if current_chunk:
        cell_type = "code" if in_code else "markdown"
        cell: dict[str, Any] = {
            "cell_type": cell_type,
            "metadata": {},
            "source": [l + "\n" for l in current_chunk],
        }
        if cell_type == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cells.append(cell)

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_notebook(
    path: str | Path,
    *,
    output_format: str = "html",
    output_dir: str | Path | None = None,
    params: dict[str, Any] | None = None,
    execute: bool = True,
    timeout: int = 600,
) -> RenderResult:
    """Render a notebook to the specified output format.

    Parameters
    ----------
    path : str | Path
        Path to the notebook file.
    output_format : str
        Output format: ``html``, ``pdf``, ``docx``, ``revealjs``, etc.
    output_dir : str | Path | None
        Output directory (default: same as source).
    params : dict[str, Any] | None
        Parameters to pass (Quarto ``params`` or Papermill parameters).
    execute : bool
        If True, execute the notebook before rendering.
    timeout : int
        Render timeout in seconds.

    Returns
    -------
    RenderResult
    """
    path = Path(path).resolve()
    if not path.is_file():
        return RenderResult(
            source=str(path),
            output_path="",
            output_format=output_format,
            success=False,
            duration_seconds=0.0,
            stderr=f"File not found: {path}",
        )

    suffix = path.suffix.lower()

    if suffix in (".qmd", ".rmd"):
        return _render_quarto(
            path, output_format=output_format, output_dir=output_dir, params=params, execute=execute, timeout=timeout
        )
    elif suffix == ".ipynb":
        return _render_jupyter(
            path, output_format=output_format, output_dir=output_dir, execute=execute, timeout=timeout
        )
    else:
        return RenderResult(
            source=str(path),
            output_path="",
            output_format=output_format,
            success=False,
            duration_seconds=0.0,
            stderr=f"Unsupported format: {suffix}",
        )


def _render_quarto(
    path: Path,
    *,
    output_format: str,
    output_dir: str | Path | None,
    params: dict[str, Any] | None,
    execute: bool,
    timeout: int,
) -> RenderResult:
    """Render a Quarto document."""
    if not _quarto_available():
        return RenderResult(
            source=str(path),
            output_path="",
            output_format=output_format,
            success=False,
            duration_seconds=0.0,
            stderr="Quarto CLI not found. Install from https://quarto.org/docs/get-started/",
        )

    cmd = ["quarto", "render", str(path), "--to", output_format]
    if output_dir:
        cmd.extend(["--output-dir", str(output_dir)])
    if not execute:
        cmd.append("--no-execute")
    if params:
        for k, v in params.items():
            cmd.extend(["-P", f"{k}:{v}"])

    rc, stdout, stderr, duration = _run_command(cmd, timeout=timeout)

    # Determine output path
    out_dir = Path(output_dir) if output_dir else path.parent
    ext_map = {
        "html": ".html",
        "pdf": ".pdf",
        "docx": ".docx",
        "revealjs": ".html",
        "pptx": ".pptx",
        "gfm": ".md",
        "epub": ".epub",
    }
    expected_ext = ext_map.get(output_format, f".{output_format}")
    output_path = out_dir / (path.stem + expected_ext)

    return RenderResult(
        source=str(path),
        output_path=str(output_path),
        output_format=output_format,
        success=rc == 0,
        duration_seconds=duration,
        stdout=stdout,
        stderr=stderr,
        return_code=rc,
    )


def _render_jupyter(
    path: Path,
    *,
    output_format: str,
    output_dir: str | Path | None,
    execute: bool,
    timeout: int,
) -> RenderResult:
    """Render a Jupyter notebook via nbconvert."""
    format_map = {
        "html": "html",
        "pdf": "pdf",
        "docx": "asciidoc",  # no direct docx; use asciidoc as fallback
        "latex": "latex",
    }
    nbconvert_format = format_map.get(output_format, output_format)

    cmd = ["jupyter", "nbconvert", "--to", nbconvert_format, str(path)]
    if execute:
        cmd.append("--execute")
    if output_dir:
        cmd.extend(["--output-dir", str(output_dir)])

    rc, stdout, stderr, duration = _run_command(cmd, timeout=timeout)

    out_dir = Path(output_dir) if output_dir else path.parent
    ext_map = {"html": ".html", "pdf": ".pdf", "latex": ".tex", "asciidoc": ".asciidoc"}
    expected_ext = ext_map.get(nbconvert_format, f".{nbconvert_format}")
    output_path = out_dir / (path.stem + expected_ext)

    return RenderResult(
        source=str(path),
        output_path=str(output_path),
        output_format=output_format,
        success=rc == 0,
        duration_seconds=duration,
        stdout=stdout,
        stderr=stderr,
        return_code=rc,
    )


def render_all(
    directory: str | Path = ".",
    *,
    output_format: str = "html",
    output_dir: str | Path | None = None,
    recursive: bool = True,
    timeout_per_notebook: int = 600,
) -> list[RenderResult]:
    """Render all notebooks in a directory.

    Parameters
    ----------
    directory : str | Path
        Directory containing notebooks.
    output_format : str
        Target format.
    output_dir : str | Path | None
        Output directory for all rendered files.
    recursive : bool
        Search subdirectories.
    timeout_per_notebook : int
        Per-notebook timeout.

    Returns
    -------
    list[RenderResult]
    """
    nbs = list_notebooks(directory, recursive=recursive)
    results: list[RenderResult] = []

    for i, nb in enumerate(nbs, 1):
        logger.info("Rendering [%d/%d]: %s", i, len(nbs), nb.filename)
        r = render_notebook(
            nb.path,
            output_format=output_format,
            output_dir=output_dir,
            timeout=timeout_per_notebook,
        )
        results.append(r)

    succeeded = sum(1 for r in results if r.success)
    logger.info("Rendered %d/%d notebooks successfully", succeeded, len(results))
    return results


def render_preview(
    path: str | Path,
    *,
    max_cells: int = 5,
    output_format: str = "html",
    timeout: int = 120,
) -> RenderResult:
    """Quick-render a notebook using only the first N cells.

    Creates a temporary copy with truncated cells and renders it.

    Parameters
    ----------
    path : str | Path
        Notebook path.
    max_cells : int
        Maximum number of cells to include.
    output_format : str
        Output format.
    timeout : int
        Render timeout.

    Returns
    -------
    RenderResult
    """
    path = Path(path).resolve()
    if not path.is_file():
        return RenderResult(
            source=str(path),
            output_path="",
            output_format=output_format,
            success=False,
            duration_seconds=0.0,
            stderr="File not found",
        )

    import tempfile

    if path.suffix.lower() == ".ipynb":
        nb = json.loads(path.read_text(encoding="utf-8"))
        nb["cells"] = nb.get("cells", [])[:max_cells]
        _fd, _tmp_name = tempfile.mkstemp(suffix=".ipynb")
        os.close(_fd)
        tmp = Path(_tmp_name)
        tmp.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    elif path.suffix.lower() in (".qmd", ".rmd"):
        content = path.read_text(encoding="utf-8")
        # Truncate after max_cells code chunks
        lines = content.splitlines()
        chunk_count = 0
        cut_line = len(lines)
        for i, line in enumerate(lines):
            if line.strip().startswith("```{"):
                chunk_count += 1
            if chunk_count > max_cells:
                cut_line = i
                break
        truncated = "\n".join(lines[:cut_line])
        _fd, _tmp_name = tempfile.mkstemp(suffix=path.suffix)
        os.close(_fd)
        tmp = Path(_tmp_name)
        tmp.write_text(truncated, encoding="utf-8")
    else:
        return RenderResult(
            source=str(path),
            output_path="",
            output_format=output_format,
            success=False,
            duration_seconds=0.0,
            stderr=f"Unsupported format: {path.suffix}",
        )

    try:
        result = render_notebook(tmp, output_format=output_format, timeout=timeout)
        result.source = str(path)  # Report original path
        return result
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def execute_notebook(
    path: str | Path,
    *,
    output_path: str | Path | None = None,
    params: dict[str, Any] | None = None,
    kernel: str = "python3",
    timeout: int = 600,
) -> RenderResult:
    """Execute a notebook non-interactively without rendering output.

    For ipynb files, uses ``jupyter nbconvert --execute --inplace``.
    For qmd files, uses ``quarto render`` with execute-only.

    Parameters
    ----------
    path : str | Path
        Notebook to execute.
    output_path : str | Path | None
        Where to write the executed notebook (default: in-place for ipynb).
    params : dict[str, Any] | None
        Execution parameters.
    kernel : str
        Kernel name for Jupyter execution.
    timeout : int
        Execution timeout.

    Returns
    -------
    RenderResult
    """
    path = Path(path).resolve()

    if path.suffix.lower() == ".ipynb":
        cmd = [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=" + str(timeout),
            "--ExecutePreprocessor.kernel_name=" + kernel,
        ]
        if output_path:
            cmd.extend(["--output", str(Path(output_path).resolve())])
        else:
            cmd.append("--inplace")
        cmd.append(str(path))

        rc, stdout, stderr, duration = _run_command(cmd, timeout=timeout + 30)
        out = str(output_path) if output_path else str(path)
        return RenderResult(
            source=str(path),
            output_path=out,
            output_format="notebook",
            success=rc == 0,
            duration_seconds=duration,
            stdout=stdout,
            stderr=stderr,
            return_code=rc,
        )
    else:
        # For qmd, just render to keep it simple
        return render_notebook(path, output_format="html", params=params, timeout=timeout)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_notebook(
    path: str | Path,
    *,
    timeout: int = 600,
) -> ValidationResult:
    """Validate a notebook by executing all cells and checking for errors.

    Parameters
    ----------
    path : str | Path
        Notebook to validate.
    timeout : int
        Execution timeout.

    Returns
    -------
    ValidationResult
    """
    path = Path(path).resolve()
    start = time.monotonic()

    if not path.is_file():
        return ValidationResult(
            path=str(path),
            valid=False,
            total_cells=0,
            executed_cells=0,
            failed_cells=0,
            errors=[{"cell": 0, "error": "File not found"}],
        )

    if path.suffix.lower() == ".ipynb":
        return _validate_ipynb(path, timeout=timeout)
    elif path.suffix.lower() in (".qmd", ".rmd"):
        return _validate_quarto(path, timeout=timeout)
    else:
        elapsed = time.monotonic() - start
        return ValidationResult(
            path=str(path),
            valid=False,
            total_cells=0,
            executed_cells=0,
            failed_cells=0,
            errors=[{"cell": 0, "error": f"Unsupported format: {path.suffix}"}],
            duration_seconds=elapsed,
        )


def _validate_ipynb(path: Path, *, timeout: int) -> ValidationResult:
    """Validate a Jupyter notebook."""
    import tempfile

    start = time.monotonic()
    _fd, _tmp_name = tempfile.mkstemp(suffix=".ipynb")
    os.close(_fd)
    tmp_out = Path(_tmp_name)

    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        f"--ExecutePreprocessor.timeout={timeout}",
        "--output",
        str(tmp_out),
        str(path),
    ]

    rc, stdout, stderr, duration = _run_command(cmd, timeout=timeout + 30)

    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    total = len(code_cells)

    errors: list[dict[str, Any]] = []
    if rc != 0:
        errors.append({"cell": -1, "error": stderr[:500]})

    # Check executed output for errors
    if tmp_out.exists():
        try:
            executed = json.loads(tmp_out.read_text(encoding="utf-8"))
            for i, cell in enumerate(executed.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                for output in cell.get("outputs", []):
                    if output.get("output_type") == "error":
                        errors.append(
                            {
                                "cell": i,
                                "ename": output.get("ename", ""),
                                "evalue": output.get("evalue", ""),
                            }
                        )
        except json.JSONDecodeError:
            pass
        tmp_out.unlink(missing_ok=True)

    elapsed = time.monotonic() - start
    return ValidationResult(
        path=str(path),
        valid=rc == 0 and len(errors) == 0,
        total_cells=total,
        executed_cells=total if rc == 0 else 0,
        failed_cells=len(errors),
        errors=errors,
        duration_seconds=elapsed,
    )


def _validate_quarto(path: Path, *, timeout: int) -> ValidationResult:
    """Validate a Quarto notebook by rendering it."""
    import tempfile

    start = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = render_notebook(
            path,
            output_format="html",
            output_dir=tmp_dir,
            timeout=timeout,
        )

        content = path.read_text(encoding="utf-8", errors="replace")
        chunk_count = sum(1 for line in content.splitlines() if line.strip().startswith("```{"))

        errors: list[dict[str, Any]] = []
        if not result.success:
            errors.append({"cell": -1, "error": result.stderr[:500]})

        elapsed = time.monotonic() - start
        return ValidationResult(
            path=str(path),
            valid=result.success,
            total_cells=chunk_count,
            executed_cells=chunk_count if result.success else 0,
            failed_cells=1 if not result.success else 0,
            errors=errors,
            duration_seconds=elapsed,
        )


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------


def convert_notebook(
    source: str | Path,
    target_format: str,
    *,
    output_path: str | Path | None = None,
) -> ConversionResult:
    """Convert a notebook between formats.

    Supported conversions: qmd <-> ipynb, Rmd <-> ipynb, qmd <-> Rmd.

    Parameters
    ----------
    source : str | Path
        Source notebook path.
    target_format : str
        Target format extension (without dot): ``qmd``, ``ipynb``, ``Rmd``.
    output_path : str | Path | None
        Output path. If None, uses source stem with new extension.

    Returns
    -------
    ConversionResult
    """
    source = Path(source).resolve()
    if not source.is_file():
        return ConversionResult(
            source=str(source),
            target="",
            source_format=source.suffix,
            target_format=target_format,
            success=False,
            message=f"Source not found: {source}",
        )

    src_fmt = source.suffix.lower()
    tgt_ext = f".{target_format}" if not target_format.startswith(".") else target_format
    tgt_ext = tgt_ext.lower()

    if output_path is None:
        output_path = source.with_suffix(tgt_ext)
    output_path = Path(output_path)

    try:
        if src_fmt == ".ipynb" and tgt_ext in (".qmd", ".rmd"):
            _ipynb_to_qmd(source, output_path)
        elif src_fmt in (".qmd", ".rmd") and tgt_ext == ".ipynb":
            _qmd_to_ipynb(source, output_path)
        elif src_fmt == ".qmd" and tgt_ext == ".rmd":
            _qmd_to_rmd(source, output_path)
        elif src_fmt == ".rmd" and tgt_ext == ".qmd":
            _rmd_to_qmd(source, output_path)
        elif src_fmt == tgt_ext:
            shutil.copy2(source, output_path)
        else:
            return ConversionResult(
                source=str(source),
                target=str(output_path),
                source_format=src_fmt,
                target_format=tgt_ext,
                success=False,
                message=f"Unsupported conversion: {src_fmt} -> {tgt_ext}",
            )

        return ConversionResult(
            source=str(source),
            target=str(output_path),
            source_format=src_fmt,
            target_format=tgt_ext,
            success=True,
            message="Conversion successful",
        )
    except Exception as exc:
        return ConversionResult(
            source=str(source),
            target=str(output_path),
            source_format=src_fmt,
            target_format=tgt_ext,
            success=False,
            message=str(exc),
        )


def _ipynb_to_qmd(source: Path, target: Path) -> None:
    """Convert ipynb to qmd."""
    nb = json.loads(source.read_text(encoding="utf-8"))
    lines: list[str] = ["---", 'title: "Converted Notebook"', "---", ""]

    for cell in nb.get("cells", []):
        cell_type = cell.get("cell_type", "")
        cell_source = "".join(cell.get("source", []))

        if cell_type == "markdown":
            lines.append(cell_source)
            lines.append("")
        elif cell_type == "code":
            # Detect language from kernel
            kernel = nb.get("metadata", {}).get("kernelspec", {}).get("language", "python")
            lines.append(f"```{{{kernel}}}")
            lines.append(cell_source)
            lines.append("```")
            lines.append("")

    target.write_text("\n".join(lines), encoding="utf-8")


def _qmd_to_ipynb(source: Path, target: Path) -> None:
    """Convert qmd/Rmd to ipynb."""
    content = source.read_text(encoding="utf-8")
    title = _extract_title(source) or "Converted Notebook"
    nb = _qmd_to_ipynb_content(content, title=title)
    target.write_text(json.dumps(nb, indent=2), encoding="utf-8")


def _qmd_to_rmd(source: Path, target: Path) -> None:
    """Convert qmd to Rmd (minimal syntax changes)."""
    content = source.read_text(encoding="utf-8")
    # QMD and RMD are very similar; main diff is YAML options
    # Replace `format:` with `output:` in YAML
    content = content.replace("format:", "output:", 1)
    target.write_text(content, encoding="utf-8")


def _rmd_to_qmd(source: Path, target: Path) -> None:
    """Convert Rmd to qmd (minimal syntax changes)."""
    content = source.read_text(encoding="utf-8")
    content = content.replace("output:", "format:", 1)
    target.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------


def extract_code_cells(
    path: str | Path,
    *,
    language: str | None = None,
) -> list[str]:
    """Extract code cells from a notebook.

    Parameters
    ----------
    path : str | Path
        Notebook path.
    language : str | None
        Filter by language (e.g. ``python``, ``r``). If None, return all.

    Returns
    -------
    list[str]
        Code cell contents as strings.
    """
    path = Path(path)
    cells: list[str] = []

    if path.suffix.lower() == ".ipynb":
        nb = json.loads(path.read_text(encoding="utf-8"))
        kernel_lang = nb.get("metadata", {}).get("kernelspec", {}).get("language", "python")
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            if language and kernel_lang.lower() != language.lower():
                continue
            cells.append("".join(cell.get("source", [])))

    elif path.suffix.lower() in (".qmd", ".rmd"):
        content = path.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        current_chunk: list[str] = []
        chunk_lang = ""
        in_chunk = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```{") and not in_chunk:
                in_chunk = True
                # Extract language: ```{python} or ```{r}
                lang = stripped[4:].rstrip("}").strip().split(",")[0].split()[0]
                chunk_lang = lang.lower()
                current_chunk = []
                continue
            elif stripped == "```" and in_chunk:
                if language is None or chunk_lang == language.lower():
                    cells.append("\n".join(current_chunk))
                current_chunk = []
                in_chunk = False
                continue
            if in_chunk:
                current_chunk.append(line)

    return cells


def extract_dependencies(
    path: str | Path,
) -> dict[str, list[str]]:
    """Extract Python import statements from a notebook.

    Parameters
    ----------
    path : str | Path
        Notebook path.

    Returns
    -------
    dict[str, list[str]]
        Keys: ``imports`` (module names), ``from_imports`` (module names).
    """
    import re

    code_cells = extract_code_cells(path, language="python")
    all_code = "\n".join(code_cells)

    imports: set[str] = set()
    from_imports: set[str] = set()

    for line in all_code.splitlines():
        line = line.strip()
        match_import = re.match(r"^import\s+([\w.]+)", line)
        match_from = re.match(r"^from\s+([\w.]+)\s+import", line)
        if match_import:
            imports.add(match_import.group(1).split(".")[0])
        if match_from:
            from_imports.add(match_from.group(1).split(".")[0])

    return {
        "imports": sorted(imports),
        "from_imports": sorted(from_imports),
    }


# ---------------------------------------------------------------------------
# Results injection
# ---------------------------------------------------------------------------


def inject_results(
    path: str | Path,
    results: dict[str, Any],
    *,
    output_path: str | Path | None = None,
    marker: str = "MOIRAIS_RESULT",
) -> Path:
    """Inject computed results into a notebook.

    Replaces placeholders of the form ``{{MOIRAIS_RESULT:key}}`` with values
    from the ``results`` dict.

    Parameters
    ----------
    path : str | Path
        Notebook path.
    results : dict[str, Any]
        Key-value pairs to inject.
    output_path : str | Path | None
        Output path (default: overwrite in place).
    marker : str
        Placeholder prefix (default ``MOIRAIS_RESULT``).

    Returns
    -------
    Path
        Path to the modified notebook.
    """
    path = Path(path)
    content = path.read_text(encoding="utf-8")

    for key, value in results.items():
        placeholder = "{{" + f"{marker}:{key}" + "}}"
        content = content.replace(placeholder, str(value))

    out = Path(output_path) if output_path else path
    out.write_text(content, encoding="utf-8")
    logger.info("Injected %d results into %s", len(results), out)
    return out


# ---------------------------------------------------------------------------
# Auto-generate notebook from module results
# ---------------------------------------------------------------------------


def generate_from_results(
    results_dir: str | Path,
    *,
    module_name: str = "analysis",
    output_path: str | Path | None = None,
    title: str | None = None,
) -> Path:
    """Auto-generate a Quarto notebook from module output CSVs.

    Scans a results directory for CSV files and creates a notebook with
    code cells that load and display each file.

    Parameters
    ----------
    results_dir : str | Path
        Directory containing CSV result files.
    module_name : str
        Module name for the title.
    output_path : str | Path | None
        Output notebook path. Default: ``results_dir/analysis_report.qmd``.
    title : str | None
        Notebook title. Default: auto-generated from module_name.

    Returns
    -------
    Path
        Path to the generated notebook.
    """
    results_dir = Path(results_dir).resolve()
    csv_files = sorted(results_dir.glob("*.csv"))

    if title is None:
        title = f"Results Report: {module_name}"
    if output_path is None:
        output_path = results_dir / "analysis_report.qmd"

    date = datetime.now().strftime("%Y-%m-%d")

    lines = [
        "---",
        f'title: "{title}"',
        f'date: "{date}"',
        "format:",
        "  html:",
        "    toc: true",
        "    code-fold: true",
        "    embed-resources: true",
        "execute:",
        "  warning: false",
        "---",
        "",
        "## Setup",
        "",
        "```{python}",
        "import pandas as pd",
        "import numpy as np",
        "from pathlib import Path",
        "```",
        "",
    ]

    for csv_file in csv_files:
        name = csv_file.stem.replace("_", " ").title()
        lines.extend(
            [
                f"## {name}",
                "",
                "```{python}",
                f'df = pd.read_csv("{csv_file}")',
                "df",
                "```",
                "",
            ]
        )

    if not csv_files:
        lines.extend(
            [
                "## No Results Found",
                "",
                f"No CSV files found in `{results_dir}`.",
                "",
            ]
        )

    output = Path(output_path)
    output.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Generated notebook from %d CSVs: %s", len(csv_files), output)
    return output


# ---------------------------------------------------------------------------
# Notebook metadata
# ---------------------------------------------------------------------------


def get_metadata(path: str | Path) -> dict[str, Any]:
    """Extract metadata from a notebook's YAML frontmatter or JSON.

    Parameters
    ----------
    path : str | Path
        Notebook path.

    Returns
    -------
    dict[str, Any]
        Metadata key-value pairs.
    """
    path = Path(path)
    if path.suffix.lower() == ".ipynb":
        nb = json.loads(path.read_text(encoding="utf-8"))
        return nb.get("metadata", {})

    # QMD/RMD: parse YAML frontmatter
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    yaml_lines: list[str] = []
    in_yaml = False

    for line in lines:
        if line.strip() == "---":
            if not in_yaml:
                in_yaml = True
                continue
            else:
                break
        if in_yaml:
            yaml_lines.append(line)

    metadata: dict[str, Any] = {}
    for line in yaml_lines:
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and value:
                metadata[key] = value

    return metadata


def update_metadata(
    path: str | Path,
    updates: dict[str, Any],
) -> None:
    """Update metadata fields in a notebook's YAML frontmatter.

    Parameters
    ----------
    path : str | Path
        Notebook path.
    updates : dict[str, Any]
        Key-value pairs to update.
    """
    path = Path(path)
    if path.suffix.lower() == ".ipynb":
        nb = json.loads(path.read_text(encoding="utf-8"))
        nb.setdefault("metadata", {}).update(updates)
        path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
        return

    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()

    # Find YAML block
    yaml_start = -1
    yaml_end = -1
    for i, line in enumerate(lines):
        if line.strip() == "---":
            if yaml_start == -1:
                yaml_start = i
            else:
                yaml_end = i
                break

    if yaml_start >= 0 and yaml_end > yaml_start:
        # Update existing keys or add new ones
        yaml_section = lines[yaml_start + 1 : yaml_end]
        updated_keys: set[str] = set()

        for idx, line in enumerate(yaml_section):
            for key, value in updates.items():
                if line.strip().startswith(f"{key}:"):
                    yaml_section[idx] = f'{key}: "{value}"' if isinstance(value, str) else f"{key}: {value}"
                    updated_keys.add(key)

        # Add keys not already present
        for key, value in updates.items():
            if key not in updated_keys:
                formatted = f'{key}: "{value}"' if isinstance(value, str) else f"{key}: {value}"
                yaml_section.append(formatted)

        new_lines = lines[: yaml_start + 1] + yaml_section + lines[yaml_end:]
        path.write_text("\n".join(new_lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Notebook diff
# ---------------------------------------------------------------------------


def diff_notebooks(
    path_a: str | Path,
    path_b: str | Path,
) -> dict[str, Any]:
    """Compare two notebooks and report differences.

    Parameters
    ----------
    path_a : str | Path
        First notebook.
    path_b : str | Path
        Second notebook.

    Returns
    -------
    dict[str, Any]
        Diff report with keys ``cells_added``, ``cells_removed``,
        ``cells_modified``, ``metadata_diff``.
    """
    cells_a = extract_code_cells(path_a)
    cells_b = extract_code_cells(path_b)

    meta_a = get_metadata(path_a)
    meta_b = get_metadata(path_b)

    # Simple cell-level diff
    max_len = max(len(cells_a), len(cells_b))
    modified = 0
    for i in range(min(len(cells_a), len(cells_b))):
        if cells_a[i].strip() != cells_b[i].strip():
            modified += 1

    cells_added = max(0, len(cells_b) - len(cells_a))
    cells_removed = max(0, len(cells_a) - len(cells_b))

    # Metadata diff
    all_keys = set(meta_a.keys()) | set(meta_b.keys())
    metadata_diff: dict[str, dict[str, Any]] = {}
    for key in all_keys:
        val_a = meta_a.get(key)
        val_b = meta_b.get(key)
        if val_a != val_b:
            metadata_diff[key] = {"a": val_a, "b": val_b}

    return {
        "path_a": str(path_a),
        "path_b": str(path_b),
        "cells_a": len(cells_a),
        "cells_b": len(cells_b),
        "cells_added": cells_added,
        "cells_removed": cells_removed,
        "cells_modified": modified,
        "metadata_diff": metadata_diff,
    }


# ---------------------------------------------------------------------------
# Notebook testing
# ---------------------------------------------------------------------------


def test_notebook(
    path: str | Path,
    *,
    expected_outputs: dict[str, Any] | None = None,
    timeout: int = 600,
) -> ValidationResult:
    """Run a notebook and check outputs against expected values.

    Parameters
    ----------
    path : str | Path
        Notebook to test.
    expected_outputs : dict[str, Any] | None
        Mapping of cell index (int) to expected output substring.
    timeout : int
        Execution timeout.

    Returns
    -------
    ValidationResult
    """
    result = validate_notebook(path, timeout=timeout)

    if expected_outputs and result.valid and Path(path).suffix.lower() == ".ipynb":
        nb = json.loads(Path(path).read_text(encoding="utf-8"))
        code_idx = 0
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            if code_idx in expected_outputs:
                expected = str(expected_outputs[code_idx])
                actual_outputs = cell.get("outputs", [])
                actual_text = ""
                for out in actual_outputs:
                    if "text" in out:
                        actual_text += "".join(out["text"])
                    elif "data" in out:
                        actual_text += str(out["data"])

                if expected not in actual_text:
                    result.errors.append(
                        {
                            "cell": code_idx,
                            "error": f"Expected '{expected}' not found in output",
                        }
                    )
                    result.failed_cells += 1
                    result.valid = False
            code_idx += 1

    return result


# ---------------------------------------------------------------------------
# Index / catalog generation
# ---------------------------------------------------------------------------


def generate_index(
    directory: str | Path = ".",
    *,
    output_path: str | Path | None = None,
    output_format: str = "qmd",
) -> Path:
    """Generate a notebook index/catalog page.

    Creates a Quarto document that lists all notebooks in the directory
    with their titles, formats, and links.

    Parameters
    ----------
    directory : str | Path
        Directory to index.
    output_path : str | Path | None
        Output file path (default: ``directory/notebook_index.qmd``).
    output_format : str
        Output format (``qmd`` or ``md``).

    Returns
    -------
    Path
        Path to the generated index.
    """
    catalog = build_catalog(directory, recursive=True)

    if output_path is None:
        output_path = Path(directory) / f"notebook_index.{output_format}"
    output_path = Path(output_path)

    lines = [
        "---",
        'title: "Notebook Index"',
        f'date: "{datetime.now().strftime("%Y-%m-%d")}"',
        "---",
        "",
        f"## Notebooks ({catalog.total_count} total)",
        "",
    ]

    # Group by format
    for fmt, count in sorted(catalog.by_format.items()):
        lines.append(f"### {fmt.upper()} ({count})")
        lines.append("")
        lines.append("| Notebook | Title | Cells |")
        lines.append("|---|---|---|")
        for nb in catalog.notebooks:
            if nb.format == fmt:
                title = nb.title or "(untitled)"
                lines.append(f"| [{nb.filename}]({nb.path}) | {title} | {nb.cell_count} |")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Generated notebook index: %s (%d notebooks)", output_path, catalog.total_count)
    return output_path


# ---------------------------------------------------------------------------
# Rendering display
# ---------------------------------------------------------------------------


def render_catalog(catalog: NotebookCatalog) -> None:
    """Display a notebook catalog with rich formatting.

    Parameters
    ----------
    catalog : NotebookCatalog
        Catalog to display.
    """
    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(
            title=f"Notebooks in {catalog.scan_directory}",
            box=box.ROUNDED,
            header_style="bold cyan",
        )
        table.add_column("File", style="bold")
        table.add_column("Format")
        table.add_column("Title")
        table.add_column("Cells", justify="right")
        table.add_column("Size", justify="right")

        for nb in catalog.notebooks:
            size_kb = nb.size_bytes / 1024
            size_str = f"{size_kb:.0f}K" if size_kb < 1024 else f"{size_kb / 1024:.1f}M"
            table.add_row(
                nb.filename,
                nb.format,
                nb.title or "(untitled)",
                str(nb.cell_count),
                size_str,
            )

        console.print(table)
        console.print(f"\nFormats: {catalog.by_format}")

    except ImportError:
        print(f"Notebooks in {catalog.scan_directory}: {catalog.total_count}")
        for nb in catalog.notebooks:
            print(f"  {nb.filename} ({nb.format}) - {nb.title or '(untitled)'}")


def render_batch_results(results: list[RenderResult]) -> None:
    """Display batch render results with rich formatting.

    Parameters
    ----------
    results : list[RenderResult]
        Render results to display.
    """
    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(
            title="Batch Render Results",
            box=box.ROUNDED,
            header_style="bold cyan",
        )
        table.add_column("Status", justify="center", width=6)
        table.add_column("Notebook", style="bold")
        table.add_column("Format")
        table.add_column("Duration", justify="right")

        for r in results:
            status = "[green]  OK [/green]" if r.success else "[red]FAIL[/red]"
            source_name = Path(r.source).name
            dur = f"{r.duration_seconds:.1f}s"
            table.add_row(status, source_name, r.output_format, dur)

        console.print(table)
        succeeded = sum(1 for r in results if r.success)
        console.print(f"\n{succeeded}/{len(results)} rendered successfully")

    except ImportError:
        for r in results:
            status = "OK" if r.success else "FAIL"
            print(f"  [{status}] {Path(r.source).name} ({r.duration_seconds:.1f}s)")
