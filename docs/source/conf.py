"""Sphinx configuration for MORIE developer documentation."""

from pathlib import Path
import sys

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

# ---------------------------------------------------------------------------
# Paths (industry-standard src-layout)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
_PY_PKG = _REPO / "src"
_R_PKG = _REPO / "r-package" / "morie"

# Python package on sys.path for autodoc
sys.path.insert(0, str(_PY_PKG))

# Local _ext directory on sys.path for the r_autodoc extension
sys.path.insert(0, str(_HERE))

# ---------------------------------------------------------------------------
# Project metadata — version read dynamically from pyproject.toml
# (single source of truth; no manual sync needed)
# ---------------------------------------------------------------------------
project = "MORIE"
copyright = "2026, Vansh Singh Ruhela"
author = "Vansh Singh Ruhela"
with (_REPO / "pyproject.toml").open("rb") as _fh:
    release = tomllib.load(_fh)["project"]["version"]
version = ".".join(release.split(".")[:2])  # short X.Y form

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    # Python autodoc
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    # Markdown support
    "myst_parser",
    # Mermaid class / flow diagrams (architecture page)
    "sphinxcontrib.mermaid",
    # R autodoc bridge (local extension)
    "_ext.r_autodoc",
]

# ---------------------------------------------------------------------------
# R autodoc bridge configuration
# ---------------------------------------------------------------------------
# Path to the R package root (contains DESCRIPTION, R/, man/)
r_package_dir = str(_R_PKG)

# Path to the man/ directory where Roxygen2 writes .Rd files
r_man_dir = str(_R_PKG / "man")

# ---------------------------------------------------------------------------
# autodoc / autosummary (Python)
# ---------------------------------------------------------------------------
autodoc_mock_imports = [
    # Mock ONLY heavy deps that a stripped local environment is
    # likely to be missing. NEVER mock pandas / numpy / scipy —
    # those are real classes that participate in `|` type-unions
    # (`pd.DataFrame | None`), and Sphinx's MockObject doesn't
    # implement `__or__`, so the import fails at function-def time.
    "doubleml", "DoubleML", "codecarbon",
    "sklearn", "statsmodels", "matplotlib",
    "openpyxl", "lxml",
]
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
    "inherited-members": False,
}
autodoc_typehints = "description"
autosummary_generate = True

# Suppress duplicate warnings from autosummary + automodule documenting same classes
suppress_warnings = [
    "autosectionlabel.*",
    "ref.python",  # "more than one target found for cross-reference"
    "app.add_directive",
    "autodoc",
    "autodoc.import_object",
    "ref.citation",  # unreferenced bib citations
    "docutils",  # duplicate object description
    "misc.highlighting_failure",
]

# ---------------------------------------------------------------------------
# Napoleon (NumPy + Google docstring styles)
# ---------------------------------------------------------------------------
napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
# Emit :ivar: directives for "Attributes" sections so dataclass fields
# don't get duplicate-registered (once by autodoc via type annotations,
# once by Napoleon via the docstring). This is the canonical fix.
napoleon_use_ivar = True

# ---------------------------------------------------------------------------
# intersphinx
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy":  ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "scipy":  ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "statsmodels": ("https://www.statsmodels.org/stable", None),
}

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md":  "markdown",
}
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ---------------------------------------------------------------------------
# HTML output — alabaster (matches sphinx-doc.org master)
# ---------------------------------------------------------------------------
html_theme = "alabaster"
html_static_path = ["_static"]
# Files copied verbatim to the site root (clean URLs for curl-able shell
# installers, e.g. https://hadesllm.github.io/morie/install.sh).
html_extra_path = ["_extra"]
html_title = "MORIE"

# "View page source" → resolve to the .rst on GitHub instead of a stub
# inside the build tree.  Lets readers click the link and land on a
# `Suggest edit` button without us having to wire up an external theme.
html_show_sourcelink = True
html_context = {
    "display_github": True,
    "github_user": "hadesllm",
    "github_repo": "morie",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
    "source_url_prefix": "https://github.com/hadesllm/morie/blob/main/docs/source/",
}

# Sidebar toggle — alabaster doesn't ship a collapse button; this
# adds a small fixed-position button at the top-left that hides /
# shows .sphinxsidebar.  Persists choice in localStorage.
html_css_files = [
    "sidebar-toggle.css",
    "clipboard-strip-comments.css",
    "sidebar-contrast.css",
]
html_js_files = [
    "sidebar-toggle.js",
    "clipboard-strip-comments.js",
]

# alabaster colour + layout knobs to match the official Sphinx site:
# off-white body, classic Sphinx blue (#2980B9), subtle gray sidebar,
# fixed sidebar so the toctree is always visible.
html_theme_options = {
    "description": "Reproducible scientific computing — 36k+ registered callables, 60+ built-in datasets (plus ~160 BigQuery-derived via Datasette), Python + R.",
    "github_user": "hadesllm",
    "github_repo": "morie",
    "github_button": False,
    "fixed_sidebar": True,
    "sidebar_collapse": True,
    "show_powered_by": False,
    "page_width": "1080px",
    "sidebar_width": "240px",
    "body_text": "#3E4349",
    "footer_text": "#888",
    "link": "#2980B9",
    "link_hover": "#1a5e8a",
    "sidebar_link": "#444",
    "sidebar_header": "#333",
    "anchor": "#888",
    "anchor_hover_fg": "#333",
    # Narrow-sidebar (mobile / zoomed-in viewport) colours — explicit so
    # captions like "Table of Contents", "Navigation", "Documentation",
    # "Development" stay readable. Alabaster's defaults render them
    # near-white on a light-blue strip and disappear at >=150% zoom.
    "narrow_sidebar_bg": "#e8eef0",
    "narrow_sidebar_fg": "#3E4349",
    "narrow_sidebar_link": "#2980B9",
    "code_bg": "#f5f5f5",
    "pre_bg": "#fafafa",
    "narrow_sidebar_bg": "#eee",
    "narrow_sidebar_link": "#444",
    "narrow_sidebar_fg": "#333",
}

html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "relations.html",
        "searchbox.html",
    ]
}

# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    }
}

