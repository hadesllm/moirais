"""Agentic loop for Perseus — MORIE's resident AI with tool calling."""

from __future__ import annotations

import io
import json
import logging
import os
import traceback
from collections.abc import Iterator
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "perseus:e2b"
_DEFAULT_BASE_URL = "http://localhost:11434"
_MAX_TOOL_OUTPUT = 8000
_TEMPERATURE = 0.1
_NUM_PREDICT = 4096


@dataclass
class AgentResponse:
    """Response from a Perseus agent loop."""

    text: str
    tool_calls_made: list[dict] = field(default_factory=list)
    iterations: int = 0
    model: str = ""


@dataclass
class _ToolCall:
    name: str
    arguments: dict


def _find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for ancestor in (here, *here.parents):
        if (ancestor / "pyproject.toml").exists():
            return ancestor
    return here


def _sanitize_path(requested: str, sandbox: Path) -> Path:
    resolved = (sandbox / requested).resolve()
    if not str(resolved).startswith(str(sandbox)):
        raise PermissionError(f"Path {resolved} is outside sandbox {sandbox}")
    return resolved


def _truncate(text: str, limit: int = _MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2 - 20
    return text[:half] + f"\n\n... [{len(text) - limit} chars truncated] ...\n\n" + text[-half:]


def tool_read_file(path: str, *, max_lines: int = 200, sandbox: Path | None = None) -> str:
    """Read a file, sandboxed to project root."""
    sandbox = sandbox or _find_project_root()
    try:
        target = _sanitize_path(path, sandbox)
    except PermissionError as exc:
        return str(exc)
    if not target.exists():
        return f"File not found: {target}"
    if not target.is_file():
        return f"Not a file: {target}"
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        if len(lines) > max_lines:
            body = "\n".join(lines[:max_lines])
            return f"{body}\n\n[... {len(lines) - max_lines} more lines truncated]"
        return "\n".join(lines)
    except Exception as exc:
        return f"Error reading {target}: {exc}"


def tool_write_file(path: str, content: str, *, sandbox: Path | None = None) -> str:
    """Write a file, sandboxed to project root."""
    sandbox = sandbox or _find_project_root()
    try:
        target = _sanitize_path(path, sandbox)
    except PermissionError as exc:
        return str(exc)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {target}"
    except Exception as exc:
        return f"Error writing {target}: {exc}"


def tool_execute_code(code: str, language: str = "python") -> str:
    """Execute code via PolyglotEngine and return output."""
    try:
        from .polyglot import PolyglotEngine
    except ImportError:
        return "PolyglotEngine not available"

    captured: list[str] = []
    auto = language != "python"
    engine = PolyglotEngine(polyglot=True, auto_detect=auto, output_fn=captured.append)
    prefix_map = {"r": "R> ", "shell": "!", "sql": "Q> "}
    prefix = prefix_map.get(language, "")
    result = engine.execute(f"{prefix}{code}" if prefix else code)

    parts: list[str] = []
    if result.stdout:
        parts.append(result.stdout)
    if result.stderr:
        parts.append(f"STDERR: {result.stderr}")
    if captured:
        parts.append("\n".join(captured))
    if not parts:
        parts.append("(no output)")
    if not result.success:
        parts.insert(0, "[EXECUTION FAILED]")
    return _truncate("\n".join(parts))


def tool_web_fetch(url: str, *, max_chars: int = 5000) -> str:
    """Fetch a URL and return text content."""
    try:
        import httpx
    except ImportError:
        return "httpx not installed"

    if not url.startswith(("http://", "https://")):
        return f"Invalid URL scheme: {url}"

    try:
        resp = httpx.get(
            url,
            timeout=15.0,
            follow_redirects=True,
            headers={
                "User-Agent": "morie-agent/1.0",
            },
        )
        resp.raise_for_status()
        text = resp.text
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[... truncated at {max_chars} chars]"
        return text
    except Exception as exc:
        return f"Fetch error: {exc}"


def tool_search_functions(query: str, *, limit: int = 10) -> str:
    """Search the fn/ registry by keyword."""
    try:
        from .fn._registry import REGISTRY
    except ImportError:
        return "Registry not available"

    query_lower = query.lower()
    tokens = query_lower.split()

    scored: list[tuple[int, str]] = []
    for key, entry in REGISTRY.items():
        searchable = f"{key} {entry.full} {entry.category} {entry.description}".lower()
        score = sum(1 for tok in tokens if tok in searchable)
        if score > 0:
            scored.append((score, key))

    scored.sort(key=lambda x: -x[0])
    results = scored[:limit]

    if not results:
        return f"No functions found matching '{query}'"

    lines: list[str] = []
    for _, key in results:
        entry = REGISTRY[key]
        lines.append(f"  {key:8s} | {entry.category:16s} | {entry.description}")

        try:
            mod = __import__(f"morie.fn.{key}", fromlist=[key])
            fn = getattr(mod, key, None)
            if fn and fn.__doc__:
                doc_lines = fn.__doc__.strip().splitlines()
                preview = doc_lines[0] if doc_lines else ""
                if preview:
                    lines.append(f"           doc: {preview[:120]}")
        except Exception:
            pass

    return "\n".join(lines)


def tool_list_files(path: str = ".", *, pattern: str = "*", sandbox: Path | None = None) -> str:
    """List files in a directory, sandboxed."""
    sandbox = sandbox or _find_project_root()
    try:
        target = _sanitize_path(path, sandbox)
    except PermissionError as exc:
        return str(exc)
    if not target.is_dir():
        return f"Not a directory: {target}"
    try:
        matches = sorted(target.glob(pattern))[:200]
        if not matches:
            return f"No files matching '{pattern}' in {target}"
        lines = []
        for p in matches:
            kind = "d" if p.is_dir() else "f"
            size = p.stat().st_size if p.is_file() else 0
            lines.append(f"  [{kind}] {p.name:40s} {size:>10d}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Error listing {target}: {exc}"


def tool_inspect_error() -> str:
    """Get the last Python traceback for debugging."""
    try:
        from .llm import get_last_traceback

        tb = get_last_traceback()
        if tb:
            return _truncate(tb)
        return "No recent Python traceback found."
    except Exception as exc:
        return f"Error inspecting traceback: {exc}"


def tool_get_cheatsheet(name: str) -> str:
    """Get the cheatsheet for an morie function (usage, signature, examples)."""
    try:
        mod = __import__(f"morie.fn.{name}", fromlist=["cheatsheet"])
        cs_fn = getattr(mod, "cheatsheet", None)
        if cs_fn:
            return cs_fn()
        fn = getattr(mod, name, None)
        if fn and fn.__doc__:
            return fn.__doc__
        return f"No cheatsheet for '{name}'"
    except Exception as exc:
        return f"Error loading cheatsheet for '{name}': {exc}"


def tool_search_codebase(pattern: str, *, file_glob: str = "*.py", max_results: int = 20, sandbox: Path | None = None) -> str:
    """Search the morie codebase for a pattern (grep-like)."""
    import re

    sandbox = sandbox or _find_project_root()
    morie_src = sandbox / "libexec" / "config" / "tools" / "py-package" / "morie"
    if not morie_src.is_dir():
        morie_src = sandbox

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return f"Invalid regex: {exc}"

    matches: list[str] = []
    for path in sorted(morie_src.rglob(file_glob)):
        if "__pycache__" in str(path):
            continue
        try:
            for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                if regex.search(line):
                    rel = path.relative_to(sandbox)
                    matches.append(f"  {rel}:{i}: {line.strip()[:120]}")
                    if len(matches) >= max_results:
                        return "\n".join(matches) + f"\n\n[... more results truncated at {max_results}]"
        except Exception:
            continue

    if not matches:
        return f"No matches for '{pattern}' in {file_glob}"
    return "\n".join(matches)


def tool_run_shell(command: str, *, timeout: int = 30) -> str:
    """Run a shell command and return output. Blocked: rm, mkfs, dd, reboot, shutdown."""
    import subprocess

    blocked = {"rm", "mkfs", "dd", "reboot", "shutdown", "poweroff", "halt", "kill", "pkill"}
    first_word = command.strip().split()[0] if command.strip() else ""
    if first_word in blocked:
        return f"Blocked: '{first_word}' is not allowed for safety."

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(_find_project_root()),
        )
        parts = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(f"STDERR: {result.stderr}")
        if result.returncode != 0:
            parts.insert(0, f"[exit code {result.returncode}]")
        return _truncate("\n".join(parts)) if parts else "(no output)"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"
    except Exception as exc:
        return f"Shell error: {exc}"


def tool_describe_data(code: str = "") -> str:
    """Describe a dataset: load from morie.data or describe a DataFrame expression."""
    stdout_buf = io.StringIO()
    try:
        exec_globals: dict[str, Any] = {}
        exec(
            "import pandas as pd; import numpy as np; "
            "from morie.data import load_dataset, DATASET_CATALOG; "
            + code,
            exec_globals,
        )
        df = exec_globals.get("df")
        if df is not None and hasattr(df, "describe"):
            with redirect_stdout(stdout_buf):
                print(f"Shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                print(f"Dtypes:\n{df.dtypes}")
                print(f"\nDescribe:\n{df.describe()}")
                print(f"\nNull counts:\n{df.isnull().sum()}")
            return _truncate(stdout_buf.getvalue())
        return stdout_buf.getvalue() or "(no DataFrame named 'df' found in output)"
    except Exception:
        return f"Error:\n{traceback.format_exc()}"


def tool_run_morie_function(name: str, kwargs: dict | None = None) -> str:
    """Run any registered morie fn/ function by name."""
    try:
        from .fn._registry import REGISTRY
    except ImportError:
        return "Registry not available"

    if name not in REGISTRY:
        return f"Function '{name}' not in registry. Use search_functions to find available functions."

    kwargs = kwargs or {}

    try:
        mod = __import__(f"morie.fn.{name}", fromlist=[name])
        fn = getattr(mod, name)
    except Exception as exc:
        return f"Error importing morie.fn.{name}: {exc}"

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            result = fn(**kwargs)
    except Exception:
        return f"Error calling {name}(**{kwargs}):\n{traceback.format_exc()}"

    parts: list[str] = []
    out = stdout_buf.getvalue()
    err = stderr_buf.getvalue()
    if out:
        parts.append(out)
    if err:
        parts.append(f"STDERR: {err}")

    if result is not None:
        try:
            result_str = repr(result)
        except Exception:
            result_str = str(type(result))
        parts.append(f"Return: {result_str}")

    return _truncate("\n".join(parts)) if parts else "(no output, returned None)"


_DOMAIN_KNOWLEDGE: dict[str, dict] = {
    "spatial": {
        "desc": "Spatial statistics, geostatistics, spatial econometrics",
        "books": "Schabenberger & Gotway; Armstrong Spatial Voting (2nd Ed)",
        "core": ["moran", "geary", "lisa", "getis", "krig", "idw", "gstat", "sgemp", "sar", "sgcar", "gwreg"],
        "extended": ["stacf", "stscan", "stkde", "stvar", "stgwr", "trajd", "ripk", "gwpca", "quadr",
                      "proxm", "dirvt", "lufac", "smvot", "sgok", "sguk", "sgsem", "splag"],
        "workflows": {
            "autocorrelation": "moran -> geary -> lisa -> getis (global -> local)",
            "interpolation": "gstat -> krig or idw (variogram -> predict)",
            "regression": "sar -> sgcar -> gwreg (global -> local spatial models)",
            "point_pattern": "ripk -> quadr (K-function -> quadrat test)",
            "voting": "proxm -> dirvt -> lufac -> smvot (Armstrong Ch1-6)",
        },
    },
    "causal": {
        "desc": "Causal inference, treatment effects, quasi-experimental methods",
        "books": "Imbens & Rubin; Hernan & Robins; Angrist & Pischke",
        "core": ["ate", "att", "atc", "ipw", "aipw", "dml", "plr", "irm"],
        "extended": ["did", "rdd", "iv", "pliv", "cate", "gate", "late", "g_comp",
                      "psm", "matching", "rosenbaum_bound", "evalue", "synth_control"],
        "workflows": {
            "observational": "ipw -> aipw -> dml (increasing robustness)",
            "matching": "psm -> ate/att/atc (propensity matching -> effects)",
            "quasi_experimental": "did or rdd or iv (natural experiments)",
            "sensitivity": "rosenbaum_bound -> evalue (omitted-variable bias bounds)",
            "heterogeneity": "cate -> gate (subgroup treatment effects)",
            "full_pipeline": "ipw -> aipw -> dml -> cate -> rosenbaum_bound",
        },
    },
    "biomedical": {
        "desc": "Biomedical signal processing (Rangayyan & Krishnan, 941 equations)",
        "books": "Rangayyan — Biomedical Signal Analysis (3rd Ed)",
        "core": ["bwflt", "envlp", "hlbrt", "hrvtd", "hrvfd", "eegbd", "welch", "pburg", "spgrm"],
        "extended": ["lmsaf", "wienr", "nlmsf", "rlsaf", "ecgdet", "hrvtd", "hrvfd",
                      "eogdt", "emgrt", "iemg", "gdemg"],
        "workflows": {
            "emg_analysis": "bwflt -> envlp -> emgrt -> iemg (filter -> envelope -> RMS -> integrated)",
            "hrv_analysis": "ecgdet -> hrvtd -> hrvfd (R-peaks -> time domain -> frequency)",
            "eeg_analysis": "bwflt -> eegbd -> welch (filter -> band decomposition -> power spectrum)",
            "spectral": "welch -> spgrm -> pburg (spectral estimation methods)",
        },
    },
    "psychometrics": {
        "desc": "Classical test theory, IRT, DIF, factor analysis, reliability",
        "books": "DeVellis; Embretson & Reise; Crocker & Algina",
        "core": ["crba", "mcdo", "kmo", "bart", "paran", "irt1p", "irt2p", "irt3p", "difmh"],
        "extended": ["itcor", "adel", "crel", "ave", "splhf", "idisc", "irtgr", "irtpc",
                      "irtif", "irtic", "irtab", "irtfl", "diflr", "difef", "difgn", "difag"],
        "workflows": {
            "reliability": "crba -> mcdo -> splhf (Cronbach alpha -> omega -> split-half)",
            "factor_prereqs": "kmo -> bart -> paran (adequacy -> Bartlett -> parallel analysis)",
            "irt": "irt1p -> irt2p -> irt3p -> irtin -> irtic (models -> information -> ICC)",
            "dif": "difmh -> diflr -> difef (MH -> logistic regression -> effect size)",
            "full_psychometric": "kmo -> bart -> crba -> irt2p -> difmh",
        },
    },
    "reliability": {
        "desc": "Internal consistency, inter-rater, SEM, measurement precision",
        "core": ["kr20", "kr21", "crba", "mcdo", "rglb", "rsem", "rseh", "splhf"],
        "extended": ["gl1", "gl2", "gl3", "gl4", "gl5", "gl6", "rmdc", "rmci", "rcsem", "rirr"],
        "workflows": {
            "dichotomous": "kr20 -> kr21 (binary item reliability)",
            "continuous": "crba -> mcdo -> gl1-gl6 (alpha -> omega -> Guttman lambdas)",
            "measurement_error": "rsem -> rseh -> rcsem (SEM -> SE-H -> conditional SEM)",
        },
    },
    "hypothesis": {
        "desc": "Statistical hypothesis testing (parametric and nonparametric)",
        "core": ["t2smp", "chisq", "fisher", "anova", "mw", "sw", "kw", "levene"],
        "extended": ["vr", "chewy", "jarjar", "ks", "windu", "ahsoka", "fried", "mcnem"],
        "workflows": {
            "two_groups": "sw (normality) -> levene (variance) -> t2smp or mw",
            "k_groups": "sw -> levene -> anova or kw -> post-hoc",
            "categorical": "chisq or fisher (expected counts < 5 -> Fisher exact)",
            "normality": "sw -> jarjar (Shapiro-Wilk -> Jarque-Bera)",
        },
    },
    "effect_size": {
        "desc": "Standardized effect sizes, conversions, clinical significance",
        "core": ["d", "g", "eta2", "omega2", "cramv", "phi", "cles", "cliff", "vda"],
        "extended": ["rho", "tau", "ewok", "c3po", "r2d2", "d2nnt", "d2or", "d2r",
                      "or2d", "or2r", "r2d", "r2or"],
        "workflows": {
            "two_groups": "d -> g -> cles -> cliff -> vda (Cohen -> Hedges -> probability)",
            "anova": "eta2 -> omega2 (biased -> unbiased)",
            "categorical": "cramv -> phi (Cramer's V -> phi coefficient)",
            "conversions": "d2r -> d2or -> d2nnt (d <-> r <-> OR <-> NNT)",
        },
    },
    "epidemiology": {
        "desc": "Epidemiological measures, compartmental models, rates",
        "core": ["nnt", "nnh", "ird", "irr", "or_es", "rd_es", "rr_es"],
        "extended": ["sir", "seir", "r0", "cfr", "le", "ebac", "legal"],
        "workflows": {
            "measures": "rr_es -> or_es -> rd_es -> nnt (risk -> odds -> difference -> NNT)",
            "rates": "ird -> irr (incidence rate difference -> ratio)",
            "modeling": "sir -> seir -> r0 (basic -> extended -> reproduction number)",
        },
    },
    "distributions": {
        "desc": "Probability distributions: density, CDF, quantile, random",
        "core": ["dnorm", "pnorm", "qnorm", "rnorm", "dt", "pt", "qt", "rt"],
        "extended": ["dchsq", "pchsq", "qchsq", "rchisq", "dbnm", "pbnm",
                      "dpoi", "ppoi", "dunf", "punf", "dexp", "pexp",
                      "dbet", "pbet", "dgam", "pgam"],
        "workflows": {
            "normal": "dnorm (density) -> pnorm (CDF) -> qnorm (quantile) -> rnorm (random)",
            "any_dist": "d* (density) -> p* (CDF) -> q* (quantile) -> r* (random)",
        },
    },
    "survival": {
        "desc": "Survival analysis, time-to-event, hazard models",
        "core": ["km", "lrank", "cox", "cumhz", "aft"],
        "workflows": {
            "full": "km -> lrank -> cox -> cumhz (Kaplan-Meier -> log-rank -> Cox PH -> hazards)",
        },
    },
    "ml": {
        "desc": "Machine learning: classification, regression, ensemble methods",
        "core": ["rf", "gbm", "lasso", "ridge", "cart", "svm_"],
        "extended": ["gam", "lowes", "kdesm", "isof", "adbst", "bgg", "smote", "robust", "stdb"],
        "workflows": {
            "classification": "smote (balance) -> rf or gbm -> kylo (cross-validate)",
            "regression": "lasso -> ridge -> gam (linear -> nonlinear)",
            "ensemble": "bag -> rf -> gbm -> adab (single -> ensemble -> boosted)",
            "anomaly": "isofo -> ackbar (isolation forest -> outlier detection)",
        },
    },
    "crypto": {
        "desc": "Post-quantum cryptography, lattice-based, educational",
        "books": "NIST FIPS 203/204; Regev 2005; Micciancio & Regev",
        "core": ["mlkem", "cpoly"],
        "extended": ["lwe", "rlwe", "lll", "babai", "gso", "bkz", "svpap", "lweke",
                      "hamcd", "goppa", "ldpcd", "ldpce", "lamp", "lampv", "wots", "wotsv",
                      "mktre", "xmss", "mldsa", "mldss", "mldsv", "ntru", "ntruc", "ntrud", "mcelc"],
        "workflows": {
            "key_exchange": "mlkem (ML-KEM-768 keygen -> encaps -> decaps)",
            "lattice": "lwe -> rlwe -> lweke (sample -> ring -> key exchange)",
            "reduction": "gso -> lll -> bkz (Gram-Schmidt -> LLL -> BKZ)",
            "signatures": "mldsa -> mldss -> mldsv (keygen -> sign -> verify)",
            "hash_based": "lamp -> lampv (Lamport) or wots -> wotsv (WOTS)",
            "codes": "hamcd -> goppa -> ldpce/ldpcd (Hamming -> Goppa -> LDPC)",
        },
    },
    "genomics": {
        "desc": "Population genetics, GWAS, genetic diversity",
        "core": ["gc", "hw", "fst", "tajd", "ld", "maf", "gwas1", "prs"],
        "workflows": {
            "diversity": "gc -> hw -> fst -> tajd (GC content -> HW equil -> differentiation -> selection)",
            "association": "maf -> ld -> gwas -> prs (MAF -> linkage -> association -> risk score)",
        },
    },
    "survey": {
        "desc": "Survey-weighted statistics, complex sampling, design effects",
        "core": ["srs", "strat", "clust", "deff", "cal_wg", "srvey"],
        "extended": ["hajek", "ht_tot", "ess", "ratio", "pstrat", "ps_wgt", "cglm", "jksmp"],
        "workflows": {
            "design": "srs or strat or clust -> deff -> fpc (design -> effect -> correction)",
            "estimation": "wt -> svymn -> svyse -> svyci (weight -> mean -> SE -> CI)",
        },
    },
    "regression": {
        "desc": "Linear, logistic, quantile, robust regression",
        "core": ["rey", "rey_lg", "qreg", "prbit"],
        "extended": ["tobit", "trunc", "heckm", "rey_ps", "negbn", "rey_zp"],
        "workflows": {
            "continuous": "ols -> r2d2 -> vif -> durbin (fit -> R-sq -> multicollinearity -> autocorrelation)",
            "binary": "logit or probit -> diagnostic",
            "count": "poiss -> negbn -> zinfl (Poisson -> NegBin -> zero-inflated)",
        },
    },
    "time_series": {
        "desc": "Time series analysis, forecasting, decomposition",
        "core": ["arima", "acf", "pacf", "stl", "adf", "kpss"],
        "extended": ["varfit", "vecm", "garch", "ewma"],
        "workflows": {
            "modeling": "acf -> pacf -> adf -> arima (correlogram -> stationarity -> model)",
            "decomposition": "stl (trend + seasonal + residual)",
        },
    },
    "bayesian": {
        "desc": "Bayesian inference, MCMC, conjugate analysis",
        "core": ["bpost", "bmcmc", "gibbs", "mh_"],
        "workflows": {
            "analysis": "bpost (conjugate prior) or bmcmc -> gibbs (sampling)",
        },
    },
    "otis": {
        "desc": "Ontario correctional restrictive confinement analysis",
        "core": ["rpl", "astc", "vol", "rct", "otd", "oml"],
        "extended": ["rpl_r", "rpl_gt", "rprat", "rpdur", "rpfrq", "rpfst", "rpgap",
                      "alrt1", "alrt2", "alrt3", "alco", "altm", "aldur", "alesc",
                      "vol_r", "vol_a", "vol_t"],
        "workflows": {
            "placement": "rpl -> rpl_r -> rprat -> rpdur -> rpfrq (regional -> rate -> duration -> frequency)",
            "alerts": "alrt1 -> alrt2 -> alrt3 -> alco -> altm (types -> co-occurrence -> timing)",
            "volatility": "vol -> vol_r -> vol_a -> vol_t (overall -> regional -> alert -> temporal)",
            "full": "rpl -> astc -> vol -> rct -> otd -> oml (placement -> state -> volatility -> recidivism -> duration -> ML)",
        },
    },
    "data_exploration": {
        "desc": "Data summarization, profiling, missing data, outliers",
        "core": ["luke", "leia", "finn", "grogu", "han", "ackbar"],
        "workflows": {
            "full_eda": "luke -> leia -> han -> ackbar -> finn -> grogu (summary -> profile -> missing -> outliers -> patterns -> grouped)",
        },
    },
    "wavelets": {
        "desc": "Wavelet transforms, denoising, multiresolution analysis",
        "core": ["dwtfn", "idwtf", "swtfn", "modwt", "csdnt", "wvmra"],
        "extended": ["cwtfn", "xwvlt", "sqsgm", "wvpkt"],
        "workflows": {
            "denoising": "dwtfn -> csdnt -> idwtf (decompose -> denoise -> reconstruct)",
            "analysis": "wvmra or cwtfn -> xwt (multiresolution or cross-wavelet)",
        },
    },
}

_KEYWORD_TO_DOMAIN: dict[str, list[str]] = {
    "spatial": ["spatial", "geographic", "geostat", "moran", "autocorrelation", "kriging",
                "gwr", "coordinates", "lat", "lon", "map", "semivariogram", "interpolat"],
    "causal": ["causal", "treatment", "effect", "ipw", "propensity", "dml", "counterfactual",
               "ate", "intervention", "rct", "observational", "confound"],
    "biomedical": ["signal", "emg", "ecg", "eeg", "hrv", "filter", "biosignal", "waveform",
                   "frequency", "spectral", "rangayyan", "envelope", "adaptive"],
    "psychometrics": ["reliability", "validity", "irt", "dif", "factor", "cronbach", "item",
                      "scale", "measurement", "psychometric", "omega"],
    "epidemiology": ["disease", "incidence", "prevalence", "nnt", "risk ratio", "odds ratio",
                     "sir model", "epidemic", "outbreak", "ebac", "blood alcohol"],
    "hypothesis": ["test", "significant", "p-value", "hypothesis", "compare groups",
                   "anova", "t-test", "chi-square", "nonparametric"],
    "effect_size": ["effect size", "cohen", "hedges", "practical significance", "magnitude",
                    "clinical significance", "nnt"],
    "survival": ["survival", "time-to-event", "hazard", "kaplan", "cox", "censored", "duration"],
    "ml": ["machine learning", "classification", "prediction", "random forest",
           "gradient boost", "svm", "cross-validation", "feature"],
    "crypto": ["cryptography", "encryption", "lattice", "post-quantum", "lwe", "kem",
               "signature", "hash", "ntru", "dilithium"],
    "genomics": ["genetic", "genomic", "snp", "gwas", "allele", "population genetics",
                 "hardy-weinberg", "fst", "tajima"],
    "regression": ["regression", "linear model", "predict", "coefficient", "ols", "logistic",
                   "quantile regression"],
    "time_series": ["time series", "forecast", "arima", "trend", "seasonal", "stationarity"],
    "survey": ["survey", "weighted", "sampling", "design effect", "stratified", "cluster sample"],
    "bayesian": ["bayesian", "prior", "posterior", "mcmc", "credible interval", "gibbs"],
    "otis": ["correctional", "confinement", "restrictive", "otis", "incarceration", "placement"],
    "data_exploration": ["explore", "summary", "describe", "eda", "missing data", "outlier", "profile"],
    "wavelets": ["wavelet", "dwt", "decomposition", "multiresolution", "denoising"],
    "distributions": ["distribution", "probability", "density", "cdf", "quantile", "random sample"],
    "reliability": ["reliability", "internal consistency", "kr-20", "kr-21", "guttman", "sem"],
}

_TEXTBOOK_REFS: dict[str, str] = {
    "spatial autocorrelation": "Schabenberger & Gotway Ch.5 — Moran's I, Geary's C, LISA",
    "kriging": "Schabenberger & Gotway Ch.5.4 — Ordinary, universal, co-kriging",
    "variogram": "Schabenberger & Gotway Ch.5.3 — Variogram estimation and modeling",
    "spatial regression": "Schabenberger & Gotway Ch.6 — SAR, CAR, SEM, GWR",
    "geostatistics": "Schabenberger & Gotway Ch.5 — Geostatistical methods",
    "point pattern": "Schabenberger & Gotway Ch.7 — Ripley's K, quadrat test",
    "spatial voting": "Armstrong Ch.1-6 — Spatial models of choice and judgment",
    "ideal point": "Armstrong Ch.2 — Ideal point estimation in spatial voting",
    "emg": "Rangayyan Ch.8 — EMG signal analysis (envelope, spectrum, fatigue)",
    "ecg": "Rangayyan Ch.7 — ECG analysis (QRS detection, HRV, arrhythmia)",
    "eeg": "Rangayyan Ch.9 — EEG analysis (band decomposition, ERP)",
    "hrv": "Rangayyan Ch.7.5 — Heart rate variability (time/frequency domain)",
    "biosignal filtering": "Rangayyan Ch.3-4 — Butterworth, Chebyshev, adaptive filters",
    "spectral analysis": "Rangayyan Ch.5 — FFT, Welch, STFT",
    "wavelet": "Rangayyan Ch.6 — DWT, CWT, wavelet denoising",
    "adaptive filter": "Rangayyan Ch.4 — LMS, RLS, NLMS adaptive filters",
    "irt": "Embretson & Reise — Item Response Theory (1PL/2PL/3PL, GRM, PCM)",
    "dif": "Zumbo — Differential Item Functioning (MH, logistic regression)",
    "reliability": "DeVellis — Scale Development (alpha, omega, split-half)",
    "factor analysis": "Brown — Confirmatory Factor Analysis (KMO, Bartlett, CFA)",
    "causal inference": "Imbens & Rubin — Rubin causal model, propensity scores",
    "dml": "Chernozhukov et al. — Double Machine Learning (PLR, IRM, PLIV)",
    "did": "Angrist & Pischke — Difference-in-Differences",
    "rdd": "Angrist & Pischke — Regression Discontinuity Design",
    "iv": "Angrist & Pischke — Instrumental Variables (2SLS, LATE)",
    "matching": "Rosenbaum & Rubin — Propensity Score Matching",
    "survival": "Hosmer & Lemeshow — KM, Cox PH, log-rank test",
    "meta-analysis": "Borenstein et al. — Fixed/random effects, forest plots",
    "ml-kem": "NIST FIPS 203 — ML-KEM (Module-Lattice Key Encapsulation)",
    "ml-dsa": "NIST FIPS 204 — ML-DSA (Module-Lattice Digital Signature)",
    "lwe": "Regev 2005 — Learning With Errors lattice problem",
    "lattice": "Micciancio & Regev — LLL, BKZ, SVP, CVP",
    "ntru": "Hoffstein, Pipher, Silverman — NTRU cryptosystem",
    "correctional": "OTIS — Ontario Restrictive Confinement Patterns",
    "ebac": "CPADS/CSUS — Estimated Blood Alcohol Concentration",
}


def tool_domain_guide(domain: str) -> str:
    """Get complete guide for a statistical domain: functions, workflows, textbook refs."""
    domain_lower = domain.lower().strip()

    info = _DOMAIN_KNOWLEDGE.get(domain_lower)
    if info is None:
        for key, val in _DOMAIN_KNOWLEDGE.items():
            if domain_lower in key or domain_lower in val.get("desc", "").lower():
                info = val
                domain_lower = key
                break

    if info is None:
        available = ", ".join(sorted(_DOMAIN_KNOWLEDGE.keys()))
        return f"Unknown domain '{domain}'. Available: {available}\nOr use search_functions(query) to search by keyword."

    lines = [f"=== {domain_lower.upper()} ===", info.get("desc", "")]

    if "books" in info:
        lines.append(f"\nTextbooks: {info['books']}")

    lines.append(f"\nCore functions ({len(info.get('core', []))}):")
    for fn_name in info.get("core", []):
        try:
            from .fn._registry import REGISTRY
            entry = REGISTRY.get(fn_name)
            if entry:
                lines.append(f"  {fn_name:8s} | {entry.category:16s} | {entry.description[:55]}")
            else:
                lines.append(f"  {fn_name:8s} | (use search_functions to find)")
        except Exception:
            lines.append(f"  {fn_name}")

    extended = info.get("extended", [])
    if extended:
        lines.append(f"\nExtended ({len(extended)}): {', '.join(extended)}")

    workflows = info.get("workflows", {})
    if workflows:
        lines.append("\nWorkflows:")
        for wf_name, wf_desc in workflows.items():
            lines.append(f"  {wf_name}: {wf_desc}")

    total = len(info.get("core", [])) + len(extended)
    lines.append(f"\nTotal: {total} functions. Use get_cheatsheet(name) for any function's docs.")
    return "\n".join(lines)


def tool_recommend_analysis(question: str) -> str:
    """Given a research question, recommend specific morie functions and analysis workflow."""
    q = question.lower()
    matched: list[tuple[int, str]] = []

    for domain, keywords in _KEYWORD_TO_DOMAIN.items():
        score = sum(1 for kw in keywords if kw in q)
        if score > 0:
            matched.append((score, domain))

    matched.sort(key=lambda x: -x[0])

    if not matched:
        return (
            "Could not match to a specific domain. Try:\n"
            "  search_functions(query) — find functions by keyword\n"
            "  category_tree() — see all categories\n"
            "  domain_guide(domain) — guide for a specific field"
        )

    lines = ["=== ANALYSIS RECOMMENDATION ===\n"]

    for _, domain in matched[:3]:
        info = _DOMAIN_KNOWLEDGE.get(domain, {})
        lines.append(f"Domain: {domain.upper()} -- {info.get('desc', '')}")

        workflows = info.get("workflows", {})
        if workflows:
            wf_name, wf_desc = next(iter(workflows.items()))
            lines.append(f"  Recommended workflow ({wf_name}): {wf_desc}")

        core = info.get("core", [])[:6]
        if core:
            lines.append(f"  Start with: {', '.join(core)}")
        lines.append("")

    lines.append("Next steps:")
    lines.append("  1. get_cheatsheet(name) on recommended functions")
    lines.append("  2. run_morie_function(name, kwargs) to execute")
    lines.append("  3. domain_guide(domain) for the complete list")
    return "\n".join(lines)


def tool_category_tree() -> str:
    """Show full taxonomy of all function categories with counts and examples."""
    try:
        from .fn._registry import REGISTRY
    except ImportError:
        return "Registry not available"

    cats: dict[str, list[str]] = {}
    for key, entry in REGISTRY.items():
        cat = entry.category
        if cat not in cats:
            cats[cat] = []
        cats[cat].append(key)

    lines = [f"=== MORIE FUNCTION TAXONOMY -- {len(REGISTRY)} functions, {len(cats)} categories ===\n"]
    for cat, funcs in sorted(cats.items(), key=lambda x: -len(x[1])):
        preview = ", ".join(sorted(funcs)[:5])
        more = f" +{len(funcs)-5} more" if len(funcs) > 5 else ""
        lines.append(f"  {cat:22s} ({len(funcs):4d}) | {preview}{more}")

    lines.append("\nUse domain_guide(domain) or search_functions(query) to explore any category.")
    return "\n".join(lines)


def tool_similar_functions(name: str, limit: int = 10) -> str:
    """Find functions similar to a given one (same category, related keywords)."""
    try:
        from .fn._registry import REGISTRY
    except ImportError:
        return "Registry not available"

    entry = REGISTRY.get(name)
    if entry is None:
        return f"Function '{name}' not found. Use search_functions to find it."

    same_cat = [(k, e) for k, e in REGISTRY.items() if e.category == entry.category and k != name]

    ref_words = set(entry.description.lower().split())
    scored = []
    for k, e in same_cat:
        words = set(e.description.lower().split())
        overlap = len(ref_words & words)
        scored.append((overlap, k, e))
    scored.sort(key=lambda x: -x[0])

    lines = [f"Similar to '{name}' ({entry.category} -- {entry.description}):\n"]
    for _, k, e in scored[:limit]:
        lines.append(f"  {k:8s} | {e.description[:60]}")

    if not scored:
        lines.append("  (no similar functions in this category)")
    return "\n".join(lines)


def tool_textbook_reference(topic: str) -> str:
    """Look up which textbook and chapter covers a statistical topic."""
    t = topic.lower()
    lines: list[str] = []

    for ref_key, ref_val in _TEXTBOOK_REFS.items():
        if any(word in ref_key for word in t.split()) or t in ref_key:
            lines.append(f"  {ref_val}")

    if not lines:
        lines.append(f"No specific reference for '{topic}'.")
        lines.append("Try: spatial, emg, ecg, irt, causal, survival, crypto, wavelet, dml, matching")

    return "\n".join([f"=== TEXTBOOK REFERENCES for '{topic}' ===\n"] + lines)


def tool_run_pipeline(steps: str) -> str:
    """Run a sequence of morie functions as a pipeline. Results from step N available as $N in later steps.

    steps: JSON array like [{"fn":"moran","kwargs":{"values":[1,2,3]}}, ...]
    """
    try:
        step_list = json.loads(steps)
    except (json.JSONDecodeError, ValueError):
        return 'Invalid JSON. Format: [{"fn": "d", "kwargs": {"group1": [1,2], "group2": [3,4]}}, ...]'

    if not isinstance(step_list, list):
        return "Steps must be a JSON array."

    results: list[str] = []
    result_objects: list[Any] = []

    for i, step in enumerate(step_list):
        fn_name = step.get("fn", "")
        kwargs = step.get("kwargs", {})

        if not fn_name:
            results.append(f"Step {i+1}: ERROR -- missing 'fn'")
            result_objects.append(None)
            continue

        for k, v in list(kwargs.items()):
            if isinstance(v, str) and v.startswith("$"):
                try:
                    ref_idx = int(v[1:]) - 1
                    if 0 <= ref_idx < len(result_objects) and result_objects[ref_idx] is not None:
                        prev = result_objects[ref_idx]
                        kwargs[k] = getattr(prev, "value", prev)
                except (ValueError, IndexError):
                    pass

        try:
            mod = __import__(f"morie.fn.{fn_name}", fromlist=[fn_name])
            fn = getattr(mod, fn_name)
            r = fn(**kwargs)
            result_objects.append(r)

            if hasattr(r, "value") and r.value is not None:
                results.append(f"Step {i+1} [{fn_name}]: value={r.value}, name={getattr(r, 'name', '')}")
            elif hasattr(r, "success"):
                results.append(f"Step {i+1} [{fn_name}]: success={r.success}")
            else:
                results.append(f"Step {i+1} [{fn_name}]: {str(r)[:200]}")
        except Exception as exc:
            result_objects.append(None)
            results.append(f"Step {i+1} [{fn_name}]: ERROR -- {exc}")

    return "\n".join(results) if results else "(empty pipeline)"


def tool_compare_methods(methods: str, data_code: str = "") -> str:
    """Run multiple statistical methods on same data, compare results.

    methods: comma-separated fn/ names (e.g. "d,g,cles,cliff")
    data_code: Python setup code (e.g. "group1=[1,2,3]; group2=[4,5,6]")
    """
    method_list = [m.strip() for m in methods.split(",") if m.strip()]
    if not method_list:
        return "Provide comma-separated function names (e.g. 'd,g,cles')"

    context: dict[str, Any] = {}
    if data_code:
        try:
            import numpy as np
            context["np"] = np
            exec(data_code, context)  # noqa: S102
        except Exception as exc:
            return f"Error in data setup: {exc}"

    skip = {"np", "pd", "__builtins__"}
    kwargs = {k: v for k, v in context.items() if not k.startswith("_") and k not in skip}

    lines = [f"=== METHOD COMPARISON ({len(method_list)} methods) ===\n"]

    for fn_name in method_list:
        try:
            mod = __import__(f"morie.fn.{fn_name}", fromlist=[fn_name])
            fn = getattr(mod, fn_name)
            r = fn(**kwargs)

            if hasattr(r, "value") and hasattr(r, "name"):
                val = f"{r.value:.6f}" if isinstance(r.value, float) else str(r.value)
                lines.append(f"  {fn_name:8s} | {getattr(r, 'name', ''):30s} | {val}")
            elif hasattr(r, "success"):
                lines.append(f"  {fn_name:8s} | success={r.success}")
            else:
                lines.append(f"  {fn_name:8s} | {str(r)[:80]}")
        except Exception as exc:
            lines.append(f"  {fn_name:8s} | ERROR: {str(exc)[:60]}")

    return "\n".join(lines)


def tool_run_suite(domain: str, data_code: str = "") -> str:
    """Run a complete analysis suite for a domain using its core functions.

    domain: one of spatial, causal, hypothesis, effect_size, reliability, etc.
    data_code: Python setup code for your data (optional; some domains have demos)
    """
    info = _DOMAIN_KNOWLEDGE.get(domain.lower().strip())
    if info is None:
        available = ", ".join(sorted(_DOMAIN_KNOWLEDGE.keys()))
        return f"Unknown domain '{domain}'. Available: {available}"

    core = info.get("core", [])
    if not core:
        return f"No core functions for {domain}"

    demo_setups: dict[str, str] = {
        "effect_size": "import numpy as np; group1=np.random.default_rng(42).normal(0,1,50).tolist(); group2=np.random.default_rng(42).normal(0.5,1,50).tolist()",
        "hypothesis": "import numpy as np; group1=np.random.default_rng(42).normal(0,1,50).tolist(); group2=np.random.default_rng(42).normal(0.5,1.2,50).tolist()",
        "distributions": "x=0.0; df=10; p=0.05",
        "epidemiology": "a=30; b=70; c=20; d=80",
    }

    if not data_code:
        data_code = demo_setups.get(domain.lower(), "")

    context: dict[str, Any] = {}
    if data_code:
        try:
            import numpy as np
            context["np"] = np
            exec(data_code, context)  # noqa: S102
        except Exception as exc:
            return f"Error in data setup: {exc}"

    skip = {"np", "pd", "__builtins__"}
    kwargs = {k: v for k, v in context.items() if not k.startswith("_") and k not in skip}

    lines = [f"=== {domain.upper()} SUITE -- {len(core)} core functions ===\n"]
    ok = 0

    for fn_name in core:
        try:
            mod = __import__(f"morie.fn.{fn_name}", fromlist=[fn_name])
            fn = getattr(mod, fn_name)
            r = fn(**kwargs)
            ok += 1
            if hasattr(r, "value") and r.value is not None:
                val = f"{r.value:.6f}" if isinstance(r.value, float) else str(r.value)
                lines.append(f"  {fn_name:8s} | {getattr(r, 'name', ''):30s} | {val}")
            elif hasattr(r, "success"):
                lines.append(f"  {fn_name:8s} | success={r.success}")
            else:
                lines.append(f"  {fn_name:8s} | {str(r)[:80]}")
        except TypeError:
            lines.append(f"  {fn_name:8s} | (needs specific args -- use get_cheatsheet)")
        except Exception as exc:
            lines.append(f"  {fn_name:8s} | {str(exc)[:60]}")

    lines.append(f"\n{ok}/{len(core)} executed.")

    workflows = info.get("workflows", {})
    if workflows:
        lines.append("\nWorkflows:")
        for wf_name, wf_desc in list(workflows.items())[:3]:
            lines.append(f"  {wf_name}: {wf_desc}")

    return "\n".join(lines)


_CORE_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_functions",
            "description": (
                "Search the morie fn/ registry (5710+ functions) by keyword. "
                "Returns names, categories, descriptions. Use before run_morie_function."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords (e.g. 'chi squared', 'survival')",
                    },
                    "limit": {"type": "integer", "description": "Max results to return (default 10)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_morie_function",
            "description": ("Run any morie fn/ function by short name. Use search_functions first to find names."),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short name (e.g. 'dnorm', 'ate')",
                    },
                    "kwargs": {
                        "type": "object",
                        "description": "Keyword arguments for the function",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": ("Execute code in Python/R/Shell/SQL. Returns stdout, stderr, and status."),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to execute"},
                    "language": {
                        "type": "string",
                        "enum": ["python", "r", "shell", "sql"],
                        "description": "Language (default python)",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Path is relative to the project root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path (relative to project root or absolute)"},
                    "max_lines": {"type": "integer", "description": "Max lines to read (default 200)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": ("Write content to a file. Creates parent dirs."),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path (relative to project root or absolute)"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories. Path is relative to project root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default '.')"},
                    "pattern": {"type": "string", "description": "Glob pattern (default '*')"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch a URL and return its text content. Useful for looking up documentation or APIs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch (must start with http:// or https://)"},
                    "max_chars": {"type": "integer", "description": "Max characters to return (default 5000)"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_error",
            "description": ("Get the last Python traceback for debugging."),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cheatsheet",
            "description": "Get the usage cheatsheet for an morie function (signature, purpose, examples).",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Function short name (e.g. 'ate', 'dnorm', 'moran')"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_codebase",
            "description": "Search the morie source code for a pattern (regex). Useful for finding implementations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "file_glob": {"type": "string", "description": "File glob (default '*.py')"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command (git, ls, pip, pytest, etc). Destructive commands blocked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_data",
            "description": (
                "Load and describe a dataset. Pass Python code that produces a DataFrame named 'df'. "
                "Example: \"df = load_dataset('ocp21')\" or \"df = pd.read_csv('data.csv')\""
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code that creates a DataFrame named 'df'"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "domain_guide",
            "description": (
                "Get a complete guide for a statistical domain: all available functions, "
                "recommended workflows, and textbook references. Domains: spatial, causal, "
                "biomedical, psychometrics, reliability, hypothesis, effect_size, epidemiology, "
                "distributions, survival, ml, crypto, genomics, survey, regression, time_series, "
                "bayesian, otis, data_exploration, wavelets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "description": "Domain name (e.g. 'spatial', 'causal', 'biomedical')"},
                },
                "required": ["domain"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_analysis",
            "description": (
                "Given a research question or analysis goal, recommend specific morie functions "
                "and a step-by-step workflow. Matches keywords to domains and returns concrete function names."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Research question or analysis description"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "category_tree",
            "description": (
                "Show the full taxonomy of all function categories with counts and example functions. "
                "Quick overview of everything available in morie."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "similar_functions",
            "description": "Find functions similar to a given one (same category, related by description keywords).",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Function short name to find similar functions for"},
                    "limit": {"type": "integer", "description": "Max results (default 10)"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "textbook_reference",
            "description": "Look up which textbook and chapter covers a statistical topic (Rangayyan, Armstrong, Schabenberger, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic to look up (e.g. 'kriging', 'emg', 'irt', 'dml')"},
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_pipeline",
            "description": (
                "Run a sequence of morie functions as a pipeline. Each step's result is available as $N in later steps. "
                'Format: [{"fn":"moran","kwargs":{"values":[1,2,3]}}, {"fn":"geary","kwargs":{"values":[1,2,3]}}]'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {"type": "string", "description": "JSON array of {fn, kwargs} steps"},
                },
                "required": ["steps"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_methods",
            "description": (
                "Run multiple statistical methods on the same data and compare results side by side. "
                "Provide comma-separated function names and optional data setup code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "methods": {"type": "string", "description": "Comma-separated fn/ names (e.g. 'd,g,cles,cliff')"},
                    "data_code": {"type": "string", "description": "Python code to set up variables (e.g. 'group1=[1,2,3]; group2=[4,5,6]')"},
                },
                "required": ["methods"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_suite",
            "description": (
                "Run a complete analysis suite for a domain, executing all core functions with optional data. "
                "Domains with built-in demos: effect_size, hypothesis, distributions, epidemiology."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "description": "Domain name (e.g. 'effect_size', 'hypothesis', 'spatial')"},
                    "data_code": {"type": "string", "description": "Python code to set up data (optional; some domains have demo data)"},
                },
                "required": ["domain"],
            },
        },
    },
]

_TOOL_DISPATCH: dict[str, Any] = {
    "search_functions": lambda **kw: tool_search_functions(**kw),
    "run_morie_function": lambda **kw: tool_run_morie_function(**kw),
    "execute_code": lambda **kw: tool_execute_code(**kw),
    "read_file": lambda **kw: tool_read_file(**kw),
    "write_file": lambda **kw: tool_write_file(**kw),
    "list_files": lambda **kw: tool_list_files(**kw),
    "web_fetch": lambda **kw: tool_web_fetch(**kw),
    "inspect_error": lambda **kw: tool_inspect_error(),
    "get_cheatsheet": lambda **kw: tool_get_cheatsheet(**kw),
    "search_codebase": lambda **kw: tool_search_codebase(**kw),
    "run_shell": lambda **kw: tool_run_shell(**kw),
    "describe_data": lambda **kw: tool_describe_data(**kw),
    "domain_guide": lambda **kw: tool_domain_guide(**kw),
    "recommend_analysis": lambda **kw: tool_recommend_analysis(**kw),
    "category_tree": lambda **kw: tool_category_tree(),
    "similar_functions": lambda **kw: tool_similar_functions(**kw),
    "textbook_reference": lambda **kw: tool_textbook_reference(**kw),
    "run_pipeline": lambda **kw: tool_run_pipeline(**kw),
    "compare_methods": lambda **kw: tool_compare_methods(**kw),
    "run_suite": lambda **kw: tool_run_suite(**kw),
}

_SYSTEM_PROMPT_FULL = (
    "You are Perseus, the MORIE demigod — an autonomous scientific computing agent with "
    "mastery over 20 statistical domains and 5710+ functions.\n\n"
    "INTELLIGENCE TOOLS (use these FIRST to plan your approach):\n"
    "- domain_guide(domain): complete guide for spatial/causal/biomedical/psychometrics/etc.\n"
    "- recommend_analysis(question): get analysis plan from a research question\n"
    "- category_tree(): see all 100+ categories at a glance\n"
    "- textbook_reference(topic): find which textbook covers a method\n"
    "- similar_functions(name): find related functions\n\n"
    "EXECUTION TOOLS:\n"
    "- search_functions(query): keyword search across 5710+ functions\n"
    "- get_cheatsheet(name): usage docs, signature, examples for any function\n"
    "- run_morie_function(name, kwargs): call any function by short name\n"
    "- run_pipeline(steps): chain multiple functions as a pipeline\n"
    "- compare_methods(methods, data_code): run multiple methods side-by-side\n"
    "- run_suite(domain): run all core functions for a domain at once\n"
    "- execute_code(code): run Python/R/Shell/SQL code directly\n\n"
    "FILE & SYSTEM TOOLS:\n"
    "- read_file / write_file / list_files: sandboxed project I/O\n"
    "- search_codebase(pattern): regex search across source code\n"
    "- run_shell(command): shell commands (git, pip, etc.)\n"
    "- describe_data(code): load and profile datasets\n"
    "- web_fetch(url): retrieve documentation from URLs\n"
    "- inspect_error(): get last Python traceback\n\n"
    "DOMAINS: spatial (Schabenberger/Armstrong), causal (Imbens/Rubin), "
    "biomedical (Rangayyan 941 eqs), psychometrics (IRT/DIF/CFA), epidemiology, "
    "hypothesis, effect_size, survival, ML, crypto (post-quantum), genomics, "
    "survey, regression, time_series, bayesian, wavelets, OTIS correctional, "
    "distributions (49), reliability, data_exploration.\n\n"
    "WORKFLOW: Understand question -> domain_guide or recommend_analysis -> "
    "get_cheatsheet -> run_morie_function or run_suite -> verify -> report.\n"
    "Never guess function names. Never fabricate results. Show real tool output."
)

_SYSTEM_PROMPT_COMPACT = (
    "You are Perseus, MORIE's scientific computing demigod. ALWAYS use tools.\n"
    "FAST PATH: domain_guide(domain) or recommend_analysis(question) to plan.\n"
    "Then: search_functions(query) to find, get_cheatsheet(name) for docs, "
    "run_morie_function(name, kwargs) to execute.\n"
    "POWER TOOLS: run_pipeline (chain steps), compare_methods (side-by-side), "
    "run_suite (entire domain at once), category_tree (see everything).\n"
    "Short names: 'moran', 'ate', 'dnorm', 'd'. Never use descriptions as names."
)

_SMALL_MODELS = {"functiongemma", "smollm2", "gemma3n:e2b", "qwen3:0.6b", "llama3.2:1b"}


def _is_small_model(model: str) -> bool:
    base = model.split(":")[0] if ":" in model else model
    return base in _SMALL_MODELS or any(s in model for s in ("270m", "0.6b", "1b", "500m"))


class PerseusAgent:
    """Full agentic loop for Perseus with tool calling."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        base_url: str = _DEFAULT_BASE_URL,
        *,
        sandbox_root: str | Path | None = None,
        max_iterations: int = 10,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._sandbox = Path(sandbox_root).resolve() if sandbox_root else _find_project_root()
        self._max_iterations = max_iterations
        self._messages: list[dict[str, Any]] = []
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=300.0)
        return self._client

    def _build_system_prompt(self) -> str:
        base = _SYSTEM_PROMPT_COMPACT if _is_small_model(self._model) else _SYSTEM_PROMPT_FULL
        try:
            from .fn._registry import REGISTRY

            cats: dict[str, int] = {}
            for entry in REGISTRY.values():
                cats[entry.category] = cats.get(entry.category, 0) + 1
            summary = ", ".join(f"{cat}({n})" for cat, n in sorted(cats.items(), key=lambda x: -x[1])[:20])
            return f"{base}\n\nFunction categories: {summary}. Use search_functions to explore."
        except Exception:
            return base

    def _build_tool_definitions(self) -> list[dict]:
        if _is_small_model(self._model):
            core_names = {
                "search_functions", "run_morie_function", "get_cheatsheet", "execute_code",
                "domain_guide", "recommend_analysis", "run_pipeline", "category_tree",
            }
            return [t for t in _CORE_TOOLS if t["function"]["name"] in core_names]
        return list(_CORE_TOOLS)

    def _execute_tool(self, name: str, args: dict) -> str:
        handler = _TOOL_DISPATCH.get(name)
        if handler is None:
            return f"Unknown tool: {name}. Available: {', '.join(_TOOL_DISPATCH.keys())}"

        if name in ("read_file", "write_file", "list_files"):
            args["sandbox"] = self._sandbox

        try:
            result = handler(**args)
            return str(result)
        except Exception:
            return f"Tool {name} failed:\n{traceback.format_exc()}"

    def _send_to_ollama(self, messages: list[dict], tools: list[dict]) -> dict:
        client = self._get_client()
        payload = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": _TEMPERATURE,
                "num_predict": _NUM_PREDICT,
            },
        }
        resp = client.post(f"{self._base_url}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()

    def _parse_tool_calls(self, message: dict) -> list[_ToolCall]:
        raw = message.get("tool_calls", [])
        calls: list[_ToolCall] = []
        for tc in raw:
            fn_info = tc.get("function", {})
            name = fn_info.get("name", "")
            arguments = fn_info.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (json.JSONDecodeError, ValueError):
                    arguments = {"raw_input": arguments}
            if name:
                calls.append(_ToolCall(name=name, arguments=arguments))
        return calls

    def chat(self, message: str, *, stream: bool = False) -> AgentResponse:
        """Run a full agent loop: send message, handle tool calls, return final response."""
        tools = self._build_tool_definitions()
        system_prompt = self._build_system_prompt()

        try:
            from .llm import _retrieve_relevant_source

            rag_context = _retrieve_relevant_source(message)
            if rag_context:
                system_prompt += f"\n\nRelevant MORIE source code:\n{rag_context}"
        except Exception:
            pass

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ]

        all_tool_calls: list[dict] = []
        iteration = 0

        while iteration < self._max_iterations:
            iteration += 1
            logger.debug("Agent iteration %d/%d", iteration, self._max_iterations)

            try:
                data = self._send_to_ollama(messages, tools)
            except Exception as exc:
                logger.error("Ollama request failed on iteration %d: %s", iteration, exc)
                return AgentResponse(
                    text=f"LLM request failed: {exc}",
                    tool_calls_made=all_tool_calls,
                    iterations=iteration,
                    model=self._model,
                )

            assistant_msg = data.get("message", {})
            content = assistant_msg.get("content", "")
            tool_calls = self._parse_tool_calls(assistant_msg)

            if not tool_calls:
                return AgentResponse(
                    text=content or "(no response)",
                    tool_calls_made=all_tool_calls,
                    iterations=iteration,
                    model=self._model,
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [{"function": {"name": tc.name, "arguments": tc.arguments}} for tc in tool_calls],
                }
            )

            for tc in tool_calls:
                logger.info("Executing tool: %s(%s)", tc.name, json.dumps(tc.arguments)[:200])
                result = self._execute_tool(tc.name, tc.arguments)
                record = {"name": tc.name, "arguments": tc.arguments, "result": result[:500]}
                all_tool_calls.append(record)

                messages.append(
                    {
                        "role": "tool",
                        "content": result,
                    }
                )

        final_text = content if content else "(max iterations reached without final response)"
        return AgentResponse(
            text=final_text,
            tool_calls_made=all_tool_calls,
            iterations=iteration,
            model=self._model,
        )

    def chat_stream(self, message: str) -> Iterator[str]:
        """Streaming variant: yields text chunks, executes tools between chunks."""
        tools = self._build_tool_definitions()
        system_prompt = self._build_system_prompt()

        try:
            from .llm import _retrieve_relevant_source

            rag_context = _retrieve_relevant_source(message)
            if rag_context:
                system_prompt += f"\n\nRelevant MORIE source code:\n{rag_context}"
        except Exception:
            pass

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ]

        for _iteration in range(1, self._max_iterations + 1):
            try:
                data = self._send_to_ollama(messages, tools)
            except Exception as exc:
                yield f"\n[Agent error: {exc}]"
                return

            assistant_msg = data.get("message", {})
            content = assistant_msg.get("content", "")
            tool_calls = self._parse_tool_calls(assistant_msg)

            if content:
                yield content

            if not tool_calls:
                return

            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [{"function": {"name": tc.name, "arguments": tc.arguments}} for tc in tool_calls],
                }
            )

            for tc in tool_calls:
                yield f"\n[calling {tc.name}...]\n"
                result = self._execute_tool(tc.name, tc.arguments)
                messages.append({"role": "tool", "content": result})

        yield "\n[max iterations reached]"

    def reset(self) -> None:
        """Clear conversation history."""
        self._messages.clear()

    def close(self) -> None:
        """Clean up resources."""
        if self._client is not None:
            self._client.close()
            self._client = None


_TOOL_CALLING_MODELS = [
    "perseus:e2b",
    "functiongemma:270m",
    "functiongemma",
    "gemma4:e2b",
    "gemma4:e2b-it-q4_K_M",
    "qwen3.5:4b",
    "qwen3.5:2b",
    "qwen3.5:0.8b",
    "mistral-nemo",
    "nemotron-cascade-2",
    "llama3.2:3b",
]


def _pick_best_model(base_url: str) -> str:
    """Auto-detect the best tool-calling model from available Ollama models."""
    try:
        import httpx

        resp = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=10.0)
        resp.raise_for_status()
        installed = {m["name"] for m in resp.json().get("models", [])}

        for candidate in _TOOL_CALLING_MODELS:
            if candidate in installed:
                return candidate
            base = candidate.split(":")[0]
            matches = [m for m in installed if m.startswith(base)]
            if matches:
                return matches[0]
    except Exception:
        pass
    return _DEFAULT_MODEL


_TEXT_TOOL_PROMPT = (
    "You are Perseus, MORIE's scientific computing demigod. "
    "You command 20 statistical domains and 5710+ functions.\n\n"
    "TOOLS AVAILABLE:\n"
    "- domain_guide(domain): Complete guide for any domain (spatial/causal/biomedical/etc.)\n"
    "- recommend_analysis(question): Get analysis plan from a research question\n"
    "- search_functions(query): Search fn/ registry by keyword\n"
    "- run_morie_function(name, kwargs): Run any morie function by short name\n"
    "- get_cheatsheet(name): Get usage docs for a function\n"
    "- run_pipeline(steps): Chain multiple functions as a pipeline\n"
    "- compare_methods(methods, data_code): Run methods side-by-side\n"
    "- run_suite(domain): Run all core functions for a domain\n"
    "- execute_code(code): Run Python code and return output\n"
    "- category_tree(): See all 100+ categories\n\n"
    "TO CALL A TOOL, output EXACTLY this format on its own line:\n"
    '<tool_call>{"name": "domain_guide", "arguments": {"domain": "spatial"}}</tool_call>\n\n'
    "RULES:\n"
    "1. Start with domain_guide or recommend_analysis to plan your approach\n"
    "2. ALWAYS search before running — never guess function names\n"
    "3. Short names: 'moran', 'ate', 'dnorm', 'd' (max 7 chars)\n"
    "4. Be explicit about statistical assumptions and limitations\n"
    "5. Never fabricate results — only report what tools return"
)

_TOOL_CALL_RE = None


def _get_tool_call_re():
    global _TOOL_CALL_RE
    if _TOOL_CALL_RE is None:
        import re
        _TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    return _TOOL_CALL_RE


class FreeAPIAgent:
    """Agent that works over OllamaFreeAPI or any text-based LLM.

    Uses text-based tool calling: the system prompt teaches the model to
    output <tool_call> tags, which we parse and execute locally. This enables
    Perseus tool calling over the internet for free — no local hardware needed.
    """

    def __init__(self, *, max_iterations: int = 5) -> None:
        self._max_iterations = max_iterations
        self._sandbox = _find_project_root()

    def _send(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        """Send messages to FreeAPI and get response text."""
        from .fam import OllamaFreeAPI

        client = OllamaFreeAPI()

        if model is None:
            model = os.environ.get("moriefam", "mistral-nemo:custom")  # noqa: SIM112

        try:
            return client.chat_messages(
                messages=messages,
                model=model,
                temperature=_TEMPERATURE,
                num_predict=_NUM_PREDICT,
            )
        except Exception:
            prompt_parts = []
            for msg in messages:
                role, content = msg["role"], msg["content"]
                if role == "system":
                    prompt_parts.append(f"[System]\n{content}")
                elif role == "user":
                    prompt_parts.append(f"[User]\n{content}")
                elif role == "assistant":
                    prompt_parts.append(f"[Assistant]\n{content}")
                elif role == "tool":
                    prompt_parts.append(f"[Tool Result]\n{content}")
            full_prompt = "\n\n".join(prompt_parts) + "\n\n[Assistant]\n"
            return client.chat(
                prompt=full_prompt,
                model=model,
                temperature=_TEMPERATURE,
                num_predict=_NUM_PREDICT,
            )

    def _parse_tool_calls(self, text: str) -> list[_ToolCall]:
        pattern = _get_tool_call_re()
        calls = []
        for match in pattern.finditer(text):
            try:
                data = json.loads(match.group(1))
                name = data.get("name", "")
                args = data.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        args = {"raw_input": args}
                if name and name in _TOOL_DISPATCH:
                    calls.append(_ToolCall(name=name, arguments=args))
            except (json.JSONDecodeError, ValueError):
                continue
        return calls

    def _execute_tool(self, name: str, args: dict) -> str:
        handler = _TOOL_DISPATCH.get(name)
        if handler is None:
            return f"Unknown tool: {name}"
        if name in ("read_file", "write_file", "list_files"):
            args["sandbox"] = self._sandbox
        try:
            return str(handler(**args))
        except Exception:
            return f"Tool {name} failed:\n{traceback.format_exc()}"

    def chat(self, message: str, *, model: str | None = None) -> AgentResponse:
        """Run text-based agent loop over FreeAPI."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _TEXT_TOOL_PROMPT},
            {"role": "user", "content": message},
        ]

        all_tool_calls: list[dict] = []
        iteration = 0

        while iteration < self._max_iterations:
            iteration += 1
            try:
                response_text = self._send(messages, model=model)
            except Exception as exc:
                return AgentResponse(
                    text=f"FreeAPI request failed: {exc}",
                    tool_calls_made=all_tool_calls,
                    iterations=iteration,
                    model=model or "freeapi",
                )

            tool_calls = self._parse_tool_calls(response_text)

            if not tool_calls:
                clean = _get_tool_call_re().sub("", response_text).strip()
                return AgentResponse(
                    text=clean or response_text,
                    tool_calls_made=all_tool_calls,
                    iterations=iteration,
                    model=model or "freeapi",
                )

            messages.append({"role": "assistant", "content": response_text})

            for tc in tool_calls:
                result = self._execute_tool(tc.name, tc.arguments)
                record = {"name": tc.name, "arguments": tc.arguments, "result": result[:500]}
                all_tool_calls.append(record)
                messages.append({"role": "tool", "content": f"[{tc.name}] {result}"})

        clean = _get_tool_call_re().sub("", response_text).strip()
        return AgentResponse(
            text=clean or "(max iterations reached)",
            tool_calls_made=all_tool_calls,
            iterations=iteration,
            model=model or "freeapi",
        )

    def chat_stream(self, message: str, *, model: str | None = None) -> Iterator[str]:
        """Streaming variant — yields text, executes tools between rounds."""
        resp = self.chat(message, model=model)
        if resp.tool_calls_made:
            yield f"[{len(resp.tool_calls_made)} tools called]\n\n"
        yield resp.text

    def close(self) -> None:
        pass


class PerseusCloudAgent:
    """Agent that connects to a remote Perseus relay server.

    The relay runs on Pi (or any machine) and handles tool execution
    server-side. The client just sends questions and gets full results
    back — zero local compute needed.

    This is how Perseus becomes a cloud service: run the relay on Pi,
    expose via Cloudflare Tunnel or port forward, and anyone with the
    URL can use Perseus with full tool-calling capabilities.
    """

    def __init__(self, url: str, token: str | None = None) -> None:
        self._url = url.rstrip("/")
        self._token = token
        self._model = "cloud"

    def chat(self, message: str, **_kwargs: Any) -> AgentResponse:
        from .perseus_relay import PerseusCloudClient

        client = PerseusCloudClient(self._url, token=self._token)
        data = client.ask(message)
        return AgentResponse(
            text=data.get("text", ""),
            tool_calls_made=data.get("tool_calls", []),
            iterations=data.get("iterations", 1),
            model=data.get("model", "cloud"),
        )

    def chat_stream(self, message: str, **_kwargs: Any) -> Iterator[str]:
        resp = self.chat(message)
        if resp.tool_calls_made:
            yield f"[{len(resp.tool_calls_made)} tools called on cloud]\n\n"
        yield resp.text

    def close(self) -> None:
        pass


def create_agent(
    model: str | None = None,
    base_url: str | None = None,
    *,
    provider: str | None = None,
    cloud_url: str | None = None,
    cloud_token: str | None = None,
    **kwargs: Any,
) -> PerseusAgent | FreeAPIAgent | PerseusCloudAgent:
    """Create the best available Perseus agent.

    Priority:
    1. Explicit cloud URL — remote Perseus relay (full tools, server-side execution)
    2. Local/remote Ollama — native tool calling (fastest, most capable)
    3. FreeAPI — text-based tool calling over free community servers
    4. Fallback — local Ollama endpoint (may not be running)

    Environment variables:
    - PERSEUS_CLOUD_URL: Auto-connect to a Perseus relay
    - PERSEUS_CLOUD_TOKEN: Auth token for the relay
    - OLLAMA_HOST: Remote Ollama instance
    - MORIE_PI_HOST: Pi hostname (used to construct Ollama URL)
    """
    if cloud_url is None:
        cloud_url = os.environ.get("PERSEUS_CLOUD_URL")
    if cloud_url:
        if cloud_token is None:
            cloud_token = os.environ.get("PERSEUS_CLOUD_TOKEN")
        return PerseusCloudAgent(cloud_url, token=cloud_token)

    if base_url is None:
        base_url = os.environ.get("OLLAMA_HOST", os.environ.get("OLLAMA_BASE_URL", _DEFAULT_BASE_URL))
        if base_url == _DEFAULT_BASE_URL:
            pi_host = os.environ.get("MORIE_PI_HOST")
            if pi_host:
                host_part = pi_host.split("@")[-1] if "@" in pi_host else pi_host
                pi_url = f"http://{host_part}:11434"
                try:
                    import httpx
                    resp = httpx.get(f"{pi_url}/api/tags", timeout=3.0)
                    if resp.status_code == 200:
                        base_url = pi_url
                        logger.info("Auto-connected to Pi Ollama at %s", pi_url)
                except Exception:
                    pass

    if provider == "freeapi":
        return FreeAPIAgent(**kwargs)

    try:
        import httpx
        resp = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=5.0)
        if resp.status_code == 200:
            if model is None:
                model = _pick_best_model(base_url)
            return PerseusAgent(model=model, base_url=base_url, **kwargs)
    except Exception:
        pass

    pi_host = os.environ.get("MORIE_PI_HOST")
    if pi_host:
        host_part = pi_host.split("@")[-1] if "@" in pi_host else pi_host
        relay_url = f"http://{host_part}:8421"
        try:
            from .perseus_relay import PerseusCloudClient
            client = PerseusCloudClient(relay_url)
            if client.is_available():
                logger.info("Auto-connected to Perseus relay at %s", relay_url)
                return PerseusCloudAgent(relay_url)
        except Exception:
            pass

    try:
        from .fam import OllamaFreeAPI
        client = OllamaFreeAPI()
        if client.list_models():
            logger.info("No local Ollama — using FreeAPI with text-based tool calling")
            return FreeAPIAgent(**kwargs)
    except Exception:
        pass

    if model is None:
        model = _DEFAULT_MODEL
    return PerseusAgent(model=model, base_url=base_url, **kwargs)


def warmup(model: str | None = None, base_url: str | None = None) -> bool:
    """Pre-load the model into GPU/RAM so subsequent queries are fast.

    Sets keep_alive=-1 (infinite) so the model stays resident. Call this
    once when morie starts — after that, tool-calling queries take 2-10s
    instead of 30-120s.
    """
    try:
        import httpx

        if base_url is None:
            base_url = os.environ.get("OLLAMA_HOST", os.environ.get("OLLAMA_BASE_URL", _DEFAULT_BASE_URL))
        if model is None:
            try:
                from .llm import _ollama_model
                model = _ollama_model() or _DEFAULT_MODEL
            except Exception:
                model = _DEFAULT_MODEL

        resp = httpx.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": " ", "stream": False, "keep_alive": -1, "options": {"num_predict": 1}},
            timeout=120.0,
        )
        return resp.status_code == 200
    except Exception:
        return False
