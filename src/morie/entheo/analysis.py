"""
morie.entheo.analysis -- Consciousness-theory analysers.

Two public callables:

  - ``beautiful_loop_metric(eeg, fmri)``
      Bayne-Laukkonen integrated phenomenal-binding score. Aggregates
      ``_theory.binding_per_frame`` into a global scalar plus DMT-vs-PCB
      contrast when both conditions are present.

  - ``san_score(eeg, fmri)``
      Self-Aware Network recurrence score (Pirez, Cohen et al.).
      Aggregates ``_theory.san_recurrence_per_frame``.

Both accept either:
  (a) two numpy arrays (EEG and fMRI timecourses), OR
  (b) a subject record dict (as returned by ``load_dmt_imaging``); in
      case (b) the contrast across DMT/PCB is also reported.

v0.4.0-alpha implementations are deliberately simple stand-ins for
the rc1 psychometrically-calibrated versions; the API and the
RichResult shape are stable.
"""

from __future__ import annotations

import numpy as np

from ..fn._richresult import RichResult
from ._theory import binding_per_frame, san_recurrence_per_frame

__all__ = ["beautiful_loop_metric", "san_score"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_pair(record_or_eeg, fmri):
    """Coerce inputs into (eeg_dmt, fmri_dmt, eeg_pcb, fmri_pcb).

    If ``record_or_eeg`` is a dict-record, pull from .eeg / .fmri; if
    it's an array, treat (record_or_eeg, fmri) as DMT only (PCB None).
    """
    if isinstance(record_or_eeg, dict) and "fmri" in record_or_eeg:
        rec = record_or_eeg
        e = rec.get("eeg") or {}
        f = rec.get("fmri") or {}
        return (
            e.get("data_dmt"),
            f.get("data_dmt"),
            e.get("data_pcb"),
            f.get("data_pcb"),
        )
    return record_or_eeg, fmri, None, None


def _coerce(a) -> np.ndarray | None:
    if a is None:
        return None
    arr = np.asarray(a, dtype=np.float32)
    if arr.size == 0:
        return None
    return arr


# ---------------------------------------------------------------------------
# Public callables
# ---------------------------------------------------------------------------

def beautiful_loop_metric(eeg, fmri=None) -> RichResult:
    """Bayne-Laukkonen integrated phenomenal-binding metric.

    Scores the cross-modal coupling between EEG power envelope and
    fMRI gradient dispersion. In a record context, also reports the
    DMT-vs-PCB contrast.

    Parameters
    ----------
    eeg : array-like or dict-record
        Either an EEG channels x timepoints array, or a subject record
        from ``load_dmt_imaging()``.
    fmri : array-like or None
        Required if ``eeg`` is an array (parcels x timepoints).

    Returns
    -------
    RichResult
        ``.score``           : scalar phenomenal-binding score (DMT or single).
        ``.score_pcb``       : scalar PCB score (record context only).
        ``.contrast``        : DMT − PCB (record context only).
        ``.per_frame_dmt``   : numpy vector of per-frame scores.
        ``.per_frame_pcb``   : numpy vector or None.
    """
    e_dmt, f_dmt, e_pcb, f_pcb = _extract_pair(eeg, fmri)
    e_dmt, f_dmt, e_pcb, f_pcb = map(_coerce, (e_dmt, f_dmt, e_pcb, f_pcb))

    warnings_list: list[str] = []
    if e_dmt is None or f_dmt is None:
        warnings_list.append(
            "EEG or fMRI missing for primary condition; returning NaN score")
        return RichResult(
            title="Beautiful Loop phenomenal-binding metric",
            call="beautiful_loop_metric(<missing inputs>)",
            warnings=warnings_list,
            payload={"score": float("nan"), "per_frame_dmt": None},
        )

    pf_dmt = binding_per_frame(e_dmt, f_dmt)
    score_dmt = float(np.mean(np.abs(pf_dmt)))

    pf_pcb = None
    score_pcb = None
    contrast = None
    if e_pcb is not None and f_pcb is not None:
        pf_pcb = binding_per_frame(e_pcb, f_pcb)
        score_pcb = float(np.mean(np.abs(pf_pcb)))
        contrast = score_dmt - score_pcb

    summary = [("score_dmt", score_dmt)]
    if score_pcb is not None:
        summary.append(("score_pcb", score_pcb))
        summary.append(("contrast_dmt_minus_pcb", contrast))

    interp = (
        f"Beautiful Loop score = {score_dmt:.4f}. "
        + (
            f"DMT − PCB contrast = {contrast:+.4f}. "
            "Positive contrasts suggest dose-dependent increases in "
            "predictive integration of phenomenal binding."
            if contrast is not None else
            "No PCB condition supplied; report DMT score only."
        )
    )
    return RichResult(
        title="Beautiful Loop phenomenal-binding metric",
        call="beautiful_loop_metric(eeg, fmri)",
        summary_lines=summary,
        warnings=warnings_list,
        interpretation=interp,
        payload={
            "score": score_dmt,
            "score_dmt": score_dmt,
            "score_pcb": score_pcb,
            "contrast": contrast,
            "per_frame_dmt": pf_dmt,
            "per_frame_pcb": pf_pcb,
            "method": "Bayne-Laukkonen Beautiful Loop (v0.4.0-alpha toy)",
        },
    )


def san_score(eeg, fmri=None) -> RichResult:
    """Self-Aware Networks (Pirez) recurrence score.

    Scores the lag-1 autocorrelation of the joint EEG-fMRI state
    vector in a sliding window -- higher = stronger meta-cognitive
    recurrence. Reports DMT-vs-PCB contrast in record context.
    """
    e_dmt, f_dmt, e_pcb, f_pcb = _extract_pair(eeg, fmri)
    e_dmt, f_dmt, e_pcb, f_pcb = map(_coerce, (e_dmt, f_dmt, e_pcb, f_pcb))

    warnings_list: list[str] = []
    if e_dmt is None or f_dmt is None:
        warnings_list.append(
            "EEG or fMRI missing for primary condition; returning NaN score")
        return RichResult(
            title="Self-Aware Networks recurrence score",
            call="san_score(<missing inputs>)",
            warnings=warnings_list,
            payload={"score": float("nan"), "per_frame_dmt": None},
        )

    pf_dmt = san_recurrence_per_frame(e_dmt, f_dmt)
    score_dmt = float(np.mean(pf_dmt))

    pf_pcb = None
    score_pcb = None
    contrast = None
    if e_pcb is not None and f_pcb is not None:
        pf_pcb = san_recurrence_per_frame(e_pcb, f_pcb)
        score_pcb = float(np.mean(pf_pcb))
        contrast = score_dmt - score_pcb

    summary = [("score_dmt", score_dmt)]
    if score_pcb is not None:
        summary.append(("score_pcb", score_pcb))
        summary.append(("contrast_dmt_minus_pcb", contrast))

    interp = (
        f"SAN recurrence score = {score_dmt:.4f}. "
        + (
            f"DMT − PCB contrast = {contrast:+.4f}. "
            "Negative contrasts are consistent with the SAN prediction "
            "that DMT temporarily disrupts meta-cognitive recurrence."
            if contrast is not None else
            "No PCB condition supplied; report DMT score only."
        )
    )
    return RichResult(
        title="Self-Aware Networks recurrence score",
        call="san_score(eeg, fmri)",
        summary_lines=summary,
        warnings=warnings_list,
        interpretation=interp,
        payload={
            "score": score_dmt,
            "score_dmt": score_dmt,
            "score_pcb": score_pcb,
            "contrast": contrast,
            "per_frame_dmt": pf_dmt,
            "per_frame_pcb": pf_pcb,
            "method": "Pirez Self-Aware Networks (v0.4.0-alpha toy)",
        },
    )
