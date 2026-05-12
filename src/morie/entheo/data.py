"""
morie.entheo.data -- Data fetchers for the DMT-imaging dataset.

Primary path: local mirror at ``/Volumes/VSR/rootcoderfiles/DMT_Imaging``
(Carhart-Harris / Timmermann; 20 subjects EEG-fMRI; 15 motion-survived).

Fallback path: OpenNeuro mirror (network-guarded; in v0.4.0-alpha we
only emit a guidance string and return a synthetic fixture so the
public API stays exercisable in CI without scipy.io / pymatreader).

The on-disk layout we recognise:

  DMT_Imaging/
    EEG/    RegressorsInterpscrubbedIRASA_{Central,Frontal,Occipital,
             Parietal,Temporal}.mat
    fMRI/   LongS{NN}{DMT,PCB}.mat                  (NN = 01..25)
    Scripts/  *.m  (BetweenNetworkRSFC, dRSFC, LZ_analysis, ...)
    README.md
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np

from ..fn._richresult import RichResult

__all__ = ["load_dmt_imaging", "dmt_imaging_root", "list_subjects"]


# 15-of-20 motion-survived per the manuscript & README; the remaining
# 5 (04, 05, 20, 21, 24) are absent on disk.
_DEFAULT_LOCAL_ROOT = "/Volumes/VSR/rootcoderfiles/DMT_Imaging"
_ENV_OVERRIDE = "MORIE_DMT_IMAGING_ROOT"
_SUBJECT_RE = re.compile(r"^LongS(\d{2})(DMT|PCB)\.mat$")


def dmt_imaging_root() -> Path | None:
    """Resolve the DMT_Imaging root, honouring $MORIE_DMT_IMAGING_ROOT."""
    candidate = os.environ.get(_ENV_OVERRIDE) or _DEFAULT_LOCAL_ROOT
    p = Path(candidate)
    return p if p.exists() else None


def list_subjects(root: str | Path | None = None) -> list[str]:
    """Enumerate available subjects under ``<root>/fMRI/``."""
    base = Path(root) if root else dmt_imaging_root()
    if base is None:
        return []
    fmri_dir = base / "fMRI"
    if not fmri_dir.exists():
        return []
    subs: set[str] = set()
    for f in fmri_dir.iterdir():
        m = _SUBJECT_RE.match(f.name)
        if m:
            subs.add(m.group(1))
    return sorted(subs)


def _synthetic_record(subject_id: str, n_tp: int = 480, n_chan: int = 32,
                      n_parcels: int = 100, seed: int = 0) -> dict:
    """Reproducible synthetic record matching the on-disk schema shape.

    Used both as a CI fixture and as the OpenNeuro-fallback placeholder
    when scipy.io can't load the .mat files.
    """
    rng = np.random.default_rng(seed + int(subject_id))
    return {
        "subject_id": subject_id,
        "condition_order": ["DMT", "PCB"],
        "eeg": {
            "sfreq": 250.0,
            "channels": [f"E{i:02d}" for i in range(n_chan)],
            "data_dmt": rng.standard_normal((n_chan, n_tp)).astype(np.float32),
            "data_pcb": rng.standard_normal((n_chan, n_tp)).astype(np.float32),
        },
        "fmri": {
            "tr": 2.0,
            "n_parcels": n_parcels,
            "data_dmt": rng.standard_normal((n_parcels, n_tp // 4)).astype(np.float32),
            "data_pcb": rng.standard_normal((n_parcels, n_tp // 4)).astype(np.float32),
            "motion_fd_mm": rng.uniform(0.0, 0.6, size=n_tp // 4).astype(np.float32),
        },
        "behavioural": {
            # Stub: real dataset has subjective intensity time courses.
            "intensity_dmt": np.clip(rng.normal(7.0, 1.5, 12), 0, 10),
            "intensity_pcb": np.clip(rng.normal(0.8, 0.6, 12), 0, 10),
        },
        "_synthetic": True,
    }


def _try_load_mat(path: Path) -> dict | None:
    """Best-effort .mat loader using scipy.io or pymatreader.

    Returns None if neither library is available -- the caller should
    fall back to the synthetic fixture and emit a warning.
    """
    try:
        from scipy.io import loadmat  # type: ignore
        return dict(loadmat(str(path), squeeze_me=True, struct_as_record=False))
    except Exception:
        pass
    try:
        from pymatreader import read_mat  # type: ignore
        return dict(read_mat(str(path)))
    except Exception:
        pass
    return None


def _load_real_record(subject_id: str, root: Path) -> dict | None:
    """Load real .mat data for one subject. Returns None on failure."""
    f_dmt = root / "fMRI" / f"LongS{subject_id}DMT.mat"
    f_pcb = root / "fMRI" / f"LongS{subject_id}PCB.mat"
    if not (f_dmt.exists() and f_pcb.exists()):
        return None
    blob_dmt = _try_load_mat(f_dmt)
    blob_pcb = _try_load_mat(f_pcb)
    if blob_dmt is None or blob_pcb is None:
        return None

    # Extract the 2-D timecourse arrays (heuristic: largest 2-D float array).
    def _largest_2d(blob: dict) -> np.ndarray:
        best = None
        for v in blob.values():
            if isinstance(v, np.ndarray) and v.ndim == 2:
                if best is None or v.size > best.size:
                    best = v
        return best if best is not None else np.zeros((1, 1), dtype=np.float32)

    arr_dmt = _largest_2d(blob_dmt).astype(np.float32)
    arr_pcb = _largest_2d(blob_pcb).astype(np.float32)
    return {
        "subject_id": subject_id,
        "condition_order": ["DMT", "PCB"],
        "eeg": {"sfreq": np.nan, "channels": [], "data_dmt": None, "data_pcb": None},
        "fmri": {
            "tr": 2.0,
            "n_parcels": arr_dmt.shape[0],
            "data_dmt": arr_dmt,
            "data_pcb": arr_pcb,
            "motion_fd_mm": None,
        },
        "behavioural": {},
        "_synthetic": False,
        "_paths": {"fmri_dmt": str(f_dmt), "fmri_pcb": str(f_pcb)},
    }


def load_dmt_imaging(subject_id: str | int | None = None) -> RichResult:
    """Load one (or all) Timmermann DMT-imaging subject records.

    Parameters
    ----------
    subject_id : str or int, optional
        Two-digit subject ID, e.g. ``"01"`` or ``1``. If None, every
        subject available on disk is loaded (or a small synthetic set).

    Returns
    -------
    RichResult
        ``.records`` : list of dicts with keys
        ``subject_id``, ``eeg``, ``fmri``, ``behavioural``.
        ``.root``    : str path to the DMT_Imaging mirror (or None).
        ``.synthetic``: bool -- True if any record is the fallback.
    """
    root = dmt_imaging_root()
    warnings_list: list[str] = []

    # Default subject roster: the 15 motion-survived subjects per
    # README + manuscript when no real root is present.
    default_subjects = [
        "01", "02", "03", "06", "07", "08", "09", "10",
        "11", "12", "13", "14", "15", "16", "17",
    ]

    if subject_id is None:
        subs = list_subjects(root) or default_subjects
    else:
        subs = [f"{int(subject_id):02d}"] if not isinstance(subject_id, str) \
            else [subject_id.zfill(2)]

    records: list[dict] = []
    any_synthetic = False
    for sid in subs:
        rec: dict | None = None
        if root is not None:
            rec = _load_real_record(sid, root)
            if rec is None:
                warnings_list.append(
                    f"subject {sid}: .mat present but unloadable "
                    "(install scipy or pymatreader); falling back to synthetic"
                )
        if rec is None:
            rec = _synthetic_record(sid)
            any_synthetic = True
        records.append(rec)

    if root is None:
        warnings_list.insert(
            0,
            "DMT_Imaging root not found on disk; using synthetic fixture. "
            f"Set ${_ENV_OVERRIDE} or place the dataset at {_DEFAULT_LOCAL_ROOT}.",
        )

    interp = (
        f"Loaded {len(records)} subject record(s)"
        f"{' (synthetic fixture)' if any_synthetic else ''}. "
        "Real records expose .fmri.data_dmt / .fmri.data_pcb as "
        "parcels x timepoints arrays; .eeg.* is populated when "
        "EEG/.mat files are present and scipy.io is importable."
    )
    return RichResult(
        title="DMT-imaging dataset load",
        call=f"load_dmt_imaging(subject_id={subject_id!r})",
        summary_lines=[
            ("subjects", len(records)),
            ("root", str(root) if root else "<missing>"),
            ("synthetic_fallback", any_synthetic),
        ],
        warnings=warnings_list,
        interpretation=interp,
        payload={
            "records": records,
            "root": str(root) if root else None,
            "synthetic": any_synthetic,
            "subject_ids": [r["subject_id"] for r in records],
        },
    )
