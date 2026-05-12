"""Smoke tests for morie.entheo.analysis."""
from __future__ import annotations

import os

import numpy as np
import pytest


def _record():
    from morie.entheo.data import load_dmt_imaging
    os.environ["MORIE_DMT_IMAGING_ROOT"] = "/nope/this/does/not/exist"
    try:
        res = load_dmt_imaging(subject_id="01")
    finally:
        del os.environ["MORIE_DMT_IMAGING_ROOT"]
    return res["records"][0]


def test_beautiful_loop_returns_numeric_score():
    from morie.entheo import beautiful_loop_metric
    rec = _record()
    res = beautiful_loop_metric(rec)
    assert isinstance(res, dict)
    assert np.isfinite(res["score_dmt"])
    assert np.isfinite(res["score_pcb"])
    assert res["per_frame_dmt"].ndim == 1


def test_san_score_returns_numeric_score():
    from morie.entheo import san_score
    rec = _record()
    res = san_score(rec)
    assert isinstance(res, dict)
    assert np.isfinite(res["score_dmt"])
    assert np.isfinite(res["score_pcb"])
    assert res["per_frame_dmt"].ndim == 1


def test_beautiful_loop_array_inputs():
    from morie.entheo import beautiful_loop_metric
    rng = np.random.default_rng(7)
    eeg = rng.standard_normal((16, 400)).astype(np.float32)
    fmri = rng.standard_normal((50, 100)).astype(np.float32)
    res = beautiful_loop_metric(eeg, fmri)
    assert np.isfinite(res["score"])
    # No PCB in array-input mode.
    assert res["score_pcb"] is None


def test_san_score_array_inputs():
    from morie.entheo import san_score
    rng = np.random.default_rng(11)
    eeg = rng.standard_normal((16, 400)).astype(np.float32)
    fmri = rng.standard_normal((50, 100)).astype(np.float32)
    res = san_score(eeg, fmri)
    assert np.isfinite(res["score"])
    assert res["score_pcb"] is None


def test_handles_missing_inputs_gracefully():
    from morie.entheo import beautiful_loop_metric, san_score
    res1 = beautiful_loop_metric({"eeg": {}, "fmri": {}})
    res2 = san_score({"eeg": {}, "fmri": {}})
    assert np.isnan(res1["score"])
    assert np.isnan(res2["score"])
    assert len(res1.warnings) > 0
    assert len(res2.warnings) > 0
