"""morie.entheo_dmt -- loaders + analyses for the Timmermann 2023
DMT EEG-fMRI dataset (github.com/timmer500/DMT_Imaging).

Dataset summary
---------------
- 20 subjects, 15 motion-survived (1-3, 5-12, 14-15, 17-19, 20 in fMRI dir).
- 2 conditions per subject: DMT (drug) and PCB (placebo).
- fMRI: BOLD signal parcellated into 112 AAL regions, 840 TRs each
  (file ``LongS{NN}{DMT,PCB}.mat``, key ``BOLD_AAL``, shape (112, 840)).
- EEG: IRASA-decomposed regressors (interpolated + scrubbed) pooled
  to 5 cortical regions (Central, Frontal, Occipital, Parietal,
  Temporal).  Each region's .mat file (``RegressorsInterpscrubbedIRASA_*``)
  contains three (14, 840, 5) arrays:
    regDMT  -- 14 subjects × 840 TRs × 5 spectral bands under DMT
    regPCB  -- 14 subjects × 840 TRs × 5 spectral bands under PCB
    regdiff -- DMT − PCB difference
  The 5 bands are the IRASA-fractal-aperiodic decomposition; per the
  associated paper they index the canonical δ, θ, α, β, γ ranges.

This module exposes two layers:

  Layer 1 (loaders): :func:`load_fmri_subject`, :func:`load_eeg_region`,
  :func:`available_subjects`.

  Layer 2 (analyses, stubbed): :func:`spectral_band_power`,
  :func:`dynamic_functional_connectivity`, :func:`lz_complexity`.
  These are placeholders that raise NotImplementedError; they will
  delegate to the Rangayyan-style spectral / coherence / complexity
  primitives in morie.fn.* once those are wired up for EEG.

References
----------
Timmermann, C., Roseman, L., Schartner, M., et al. (2023).  Human
brain effects of DMT assessed via EEG-fMRI.  PNAS 120(13):
e2218949120.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import numpy as np

from .fn._richresult import RichResult

# ── Where the dataset lives ────────────────────────────────────────

# The DMT_Imaging dataset is expected at the path given by the
# MORIE_DMT_IMAGING_ROOT environment variable. Set it before
# importing this module; otherwise, the default placeholder below
# is used.
import os as _os
DATASET_ROOT = Path(_os.environ.get(
    "MORIE_DMT_IMAGING_ROOT",
    "/path/to/workspace/DMT_Imaging",
))

EEG_REGIONS = ("Central", "Frontal", "Occipital", "Parietal", "Temporal")
EEG_BANDS = ("delta", "theta", "alpha", "beta", "gamma")
Region = Literal["Central", "Frontal", "Occipital", "Parietal", "Temporal"]
Condition = Literal["DMT", "PCB"]


def _require_root() -> Path:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"DMT_Imaging not found at {DATASET_ROOT}.  Clone "
            "https://github.com/timmer500/DMT_Imaging.git or set "
            "MORIE_DMT_IMAGING_ROOT.")
    return DATASET_ROOT


def _loadmat(path: Path) -> dict:
    """Lightweight .mat loader.  Uses scipy.io.loadmat for v6/v7 files."""
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise RuntimeError("scipy is required to read DMT_Imaging "
                            ".mat files") from exc
    return loadmat(path)


# ── Layer 1: loaders ───────────────────────────────────────────────


def available_subjects() -> list[int]:
    """List the integer IDs of subjects that survive in the fMRI dir.

    Returns subject IDs sorted ascending (e.g. ``[1, 2, 3, 5, 6, 7,
    8, 9, 10, 11, 12, 14, 15, 17, 18, 19, 20]`` -- the
    motion-corrected subset).
    """
    root = _require_root()
    fmri_dir = root / "fMRI"
    if not fmri_dir.exists():
        return []
    pat = re.compile(r"^LongS(\d+)(DMT|PCB)\.mat$")
    ids: set[int] = set()
    for f in fmri_dir.iterdir():
        m = pat.match(f.name)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def load_fmri_subject(subject_id: int,
                       condition: Condition = "DMT") -> np.ndarray:
    """Load a subject's BOLD AAL parcellation under one condition.

    Returns
    -------
    np.ndarray of shape (112, 840) -- 112 AAL ROIs × 840 TRs.
    """
    root = _require_root()
    fname = f"LongS{subject_id:02d}{condition}.mat"
    path = root / "fMRI" / fname
    if not path.exists():
        raise FileNotFoundError(
            f"{fname} not in {path.parent}.  Available subjects: "
            f"{available_subjects()}")
    mat = _loadmat(path)
    if "BOLD_AAL" not in mat:
        raise KeyError(f"BOLD_AAL key missing from {fname}")
    return np.ascontiguousarray(mat["BOLD_AAL"], dtype=np.float64)


def load_eeg_region(region: Region) -> dict[str, np.ndarray]:
    """Load IRASA EEG regressors for one cortical region.

    Returns
    -------
    dict with keys ``regDMT``, ``regPCB``, ``regdiff`` each of shape
    (14 subjects, 840 TRs, 5 bands).  Bands index canonical δ, θ, α,
    β, γ in increasing-frequency order.
    """
    if region not in EEG_REGIONS:
        raise ValueError(f"region must be one of {EEG_REGIONS}; got {region!r}")
    root = _require_root()
    path = root / "EEG" / f"RegressorsInterpscrubbedIRASA_{region}.mat"
    mat = _loadmat(path)
    return {k: np.ascontiguousarray(mat[k], dtype=np.float64)
             for k in ("regDMT", "regPCB", "regdiff") if k in mat}


def dataset_overview() -> RichResult:
    """One-call introspection of the bundled DMT_Imaging dataset."""
    root = _require_root()
    subs = available_subjects()
    summary = [
        ("Dataset root", str(root)),
        ("Subjects (motion-survived)", len(subs)),
        ("Subject IDs", subs),
        ("Conditions", list(("DMT", "PCB"))),
        ("fMRI parcellation", "AAL (112 regions)"),
        ("fMRI timepoints", 840),
        ("EEG regions", list(EEG_REGIONS)),
        ("EEG bands", list(EEG_BANDS)),
        ("EEG axes", "(14 subj, 840 TRs, 5 bands)"),
    ]
    return RichResult(
        title="DMT_Imaging -- Timmermann 2023 dataset overview",
        summary_lines=summary,
        interpretation=(
            "Use load_fmri_subject(id, condition) for BOLD AAL "
            "matrices and load_eeg_region(region) for IRASA EEG "
            "regressors.  Layer-2 spectral / connectivity analyses "
            "are stubbed -- wire to morie.fn Rangayyan-style "
            "primitives in a follow-up."
        ),
        payload={"root": str(root), "n_subjects": len(subs),
                  "subject_ids": subs},
    )


# ── Layer 2: analyses (wired to morie.fn Rangayyan primitives) ───

# Canonical EEG bands in Hz, matching Timmermann 2023 §Methods and
# the standard biomedical-signal-analysis convention
# (Rangayyan & Krishnan 2024, Ch. 5).
DEFAULT_BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 0.5,  4.0),
    ("theta", 4.0,  8.0),
    ("alpha", 8.0, 13.0),
    ("beta",  13.0, 30.0),
    ("gamma", 30.0, 80.0),
)


def spectral_band_power(signal: np.ndarray, *,
                         fs: float = 200.0,
                         bands: tuple = DEFAULT_BANDS,
                         nperseg: int | None = None) -> RichResult:
    """Welch-PSD band-power decomposition for one EEG channel.

    Wraps :func:`morie.fn.psdwl.psdwl` to estimate the PSD, then
    integrates over each canonical band $(f_\\mathrm{lo}, f_\\mathrm{hi})$
    via the trapezoidal rule.  Output is the per-band power
    (absolute, in V²/Hz·Hz units of the input) plus the same
    expressed as a fraction of total broadband power.

    Parameters
    ----------
    signal : np.ndarray
        1-D EEG time series.
    fs : float
        Sampling frequency in Hz.  Default 200 Hz matches the
        Timmermann 2023 acquisition.
    bands : tuple of (name, f_lo, f_hi)
        Frequency bands.  Default = canonical δ/θ/α/β/γ.
    nperseg : int or None
        Welch segment length.  Defaults to ``min(len(signal), 4*fs)``
        -- i.e. 4-second segments at the default fs.
    """
    from .fn.psdwl import psdwl

    sig = np.asarray(signal, dtype=float).ravel()
    if sig.size < 16:
        return RichResult(title="spectral_band_power",
                          warnings=[f"signal too short ({sig.size} samples)"])
    if nperseg is None:
        nperseg = min(sig.size, max(64, int(4 * fs)))
    res = psdwl(sig, fs=fs, nperseg=nperseg)
    f = np.asarray(res.extra["frequencies"])
    psd = np.asarray(res.extra["psd"])
    total = float(np.trapezoid(psd, f))

    rows = []
    for name, lo, hi in bands:
        mask = (f >= lo) & (f <= hi)
        if mask.sum() < 2:
            absp = float("nan"); rel = float("nan")
        else:
            absp = float(np.trapezoid(psd[mask], f[mask]))
            rel = absp / total if total > 0 else float("nan")
        rows.append({"band": name, "f_lo": lo, "f_hi": hi,
                      "abs_power": round(absp, 6),
                      "rel_power": round(rel, 4)})
    summary = [(f"{r['band']} ({r['f_lo']}--{r['f_hi']} Hz)",
                  f"abs={r['abs_power']:.4g}, rel={r['rel_power']:.3f}")
                for r in rows]
    summary.append(("Total broadband power", round(total, 6)))
    return RichResult(
        title="EEG band-power decomposition (Welch)",
        summary_lines=summary,
        interpretation=(
            "Per-band power from Welch PSD integration.  Relative "
            "power sums to ≈1 across non-overlapping bands.  In "
            "DMT vs PCB contrasts, alpha-band relative power "
            "decreases and gamma increases under DMT."
        ),
        payload={"rows": rows, "total_power": total,
                  "bands": [r["band"] for r in rows],
                  "abs_power_per_band": [r["abs_power"] for r in rows],
                  "rel_power_per_band": [r["rel_power"] for r in rows]},
    )


def dynamic_functional_connectivity(bold: np.ndarray, *,
                                      window: int = 30,
                                      step: int = 5) -> RichResult:
    """Sliding-window inter-region BOLD correlation (dRSFC).

    For an AAL-parcellated BOLD matrix of shape (n_regions, n_TRs),
    compute the upper-triangular Pearson correlation matrix in each
    sliding window of ``window`` TRs advanced by ``step`` TRs.

    Mirrors the dRSFC.m Matlab script in DMT_Imaging/Scripts/.
    Returns per-window flattened upper-triangular correlation
    vectors plus the global mean / std across windows.
    """
    bold = np.asarray(bold, dtype=float)
    if bold.ndim != 2 or bold.shape[0] < 2 or bold.shape[1] < window + step:
        return RichResult(title="dynamic_functional_connectivity",
                          warnings=[f"insufficient BOLD shape {bold.shape}"])
    nr, nt = bold.shape
    iu = np.triu_indices(nr, k=1)
    starts = np.arange(0, nt - window + 1, step)
    n_windows = starts.size
    n_pairs = iu[0].size
    cube = np.empty((n_windows, n_pairs), dtype=np.float32)
    for i, s in enumerate(starts):
        seg = bold[:, s:s + window]
        # Pearson correlation between rows
        c = np.corrcoef(seg)
        cube[i] = c[iu]
    mean_per_pair = cube.mean(axis=0)
    std_per_pair = cube.std(axis=0)
    summary = [
        ("BOLD shape", f"{nr} regions × {nt} TRs"),
        ("Window / step (TR)", f"{window} / {step}"),
        ("Number of windows", n_windows),
        ("Number of region pairs", n_pairs),
        ("Mean across windows of mean correlation",
          round(float(mean_per_pair.mean()), 4)),
        ("Mean across windows of std correlation",
          round(float(std_per_pair.mean()), 4)),
    ]
    return RichResult(
        title="Dynamic resting-state functional connectivity (dRSFC)",
        summary_lines=summary,
        interpretation=(
            "Sliding-window Pearson FC mirrors Allen et al.\\ 2014 / "
            "the dRSFC.m script.  Higher std-of-correlation across "
            "windows indicates a more dynamically reconfiguring "
            "connectivity profile -- a Timmermann 2023 DMT signature."
        ),
        payload={"n_windows": n_windows, "n_pairs": n_pairs,
                  "mean_per_pair": mean_per_pair.tolist()[:50],
                  "std_per_pair": std_per_pair.tolist()[:50]},
    )


def lz_complexity(signal: np.ndarray, *,
                    threshold: float | None = None) -> RichResult:
    """Lempel-Ziv (LZ76) complexity of a binarised signal.

    Wraps :func:`morie.fn.lzcmp.lempel_ziv_complexity`.  The
    DMT-vs-PCB contrast on LZ complexity is one of Timmermann
    2023's headline findings: LZ rises under DMT, indicating
    increased neural-signal diversity.
    """
    from .fn.lzcmp import lempel_ziv_complexity

    sig = np.asarray(signal, dtype=float).ravel()
    if sig.size < 8:
        return RichResult(title="lz_complexity",
                          warnings=[f"signal too short ({sig.size})"])
    res = lempel_ziv_complexity(sig, threshold=threshold)
    # morie.fn.lzcmp returns ESRes with .estimate (raw LZ count) and
    # .extra["normalised"] (length-normalised, the headline statistic).
    raw = float(getattr(res, "estimate", float("nan")))
    normalised = float(getattr(res, "extra", {}).get(
        "normalised", float("nan")))
    used_threshold = float(getattr(res, "extra", {}).get(
        "threshold", float("nan")))
    return RichResult(
        title="Lempel-Ziv (LZ76) complexity",
        summary_lines=[
            ("Signal length", sig.size),
            ("Binarisation threshold",
              round(used_threshold, 4)),
            ("LZ raw count", round(raw, 1)),
            ("LZ normalised (length-corrected)", round(normalised, 4)),
        ],
        interpretation=(
            "Higher normalised LZ ⇒ more diverse / less compressible "
            "binary encoding.  In DMT vs PCB EEG contrasts, "
            "normalised LZ rises under DMT (Timmermann 2023 §Results)."
        ),
        payload={"lz_raw": raw, "lz_normalised": normalised,
                  "threshold": used_threshold},
    )


def analyze_subject(subject_id: int, *,
                     conditions: tuple = ("DMT", "PCB"),
                     window: int = 30, step: int = 5) -> RichResult:
    """Run all three Layer-2 analyses on one subject's BOLD data
    under each condition and return a comparison RichResult.

    For EEG analyses (band-power, LZ), use :func:`load_eeg_region`
    which returns multi-subject pooled data; this function focuses
    on the per-subject BOLD pipeline.
    """
    rows = []
    for cond in conditions:
        try:
            bold = load_fmri_subject(subject_id, cond)
        except FileNotFoundError as exc:
            rows.append({"subject": subject_id, "condition": cond,
                          "error": str(exc)})
            continue
        # Mean BOLD across regions as a "global signal" for LZ
        gs = bold.mean(axis=0)
        lz_res = lz_complexity(gs)
        dfc = dynamic_functional_connectivity(bold,
                                                window=window, step=step)
        rows.append({
            "subject": subject_id, "condition": cond,
            "lz_global_signal_raw": round(
                lz_res.payload.get("lz_raw", float("nan")), 1),
            "lz_global_signal_normalised": round(
                lz_res.payload.get("lz_normalised", float("nan")), 4),
            "n_dfc_windows": dfc.payload.get("n_windows"),
            "mean_dfc_corr": round(
                float(np.mean(dfc.payload.get("mean_per_pair", [0]))), 4),
        })
    summary = [("Subject", subject_id),
                ("Conditions evaluated",
                  [r["condition"] for r in rows if "error" not in r])]
    return RichResult(
        title=f"DMT-vs-PCB per-subject analysis -- subj {subject_id}",
        summary_lines=summary,
        interpretation=(
            "DMT-PCB within-subject contrast on global-signal LZ "
            "and mean dynamic FC.  Headline Timmermann 2023 finding: "
            "LZ rises under DMT for the majority of subjects."
        ),
        payload={"rows": rows},
    )
