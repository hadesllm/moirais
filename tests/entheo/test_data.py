"""Smoke tests for morie.entheo.data."""
from __future__ import annotations

import os

import numpy as np
import pytest


def test_dmt_imaging_root_resolves_or_none():
    from morie.entheo.data import dmt_imaging_root
    p = dmt_imaging_root()
    # Either we have the real dataset on disk, or the loader is
    # honest about it being absent.
    assert p is None or p.exists()


def test_list_subjects_returns_list():
    from morie.entheo.data import list_subjects
    subs = list_subjects()
    assert isinstance(subs, list)
    for s in subs:
        assert isinstance(s, str) and len(s) == 2 and s.isdigit()


def test_load_single_synthetic_subject():
    from morie.entheo.data import load_dmt_imaging
    # Force the synthetic fallback by pointing at a non-existent root.
    os.environ["MORIE_DMT_IMAGING_ROOT"] = "/nope/this/does/not/exist"
    try:
        res = load_dmt_imaging(subject_id="07")
    finally:
        del os.environ["MORIE_DMT_IMAGING_ROOT"]
    assert isinstance(res, dict)  # RichResult IS a dict
    assert len(res["records"]) == 1
    rec = res["records"][0]
    assert rec["subject_id"] == "07"
    assert rec["_synthetic"] is True
    eeg_dmt = rec["eeg"]["data_dmt"]
    fmri_dmt = rec["fmri"]["data_dmt"]
    assert isinstance(eeg_dmt, np.ndarray) and eeg_dmt.ndim == 2
    assert isinstance(fmri_dmt, np.ndarray) and fmri_dmt.ndim == 2


def test_load_all_synthetic_when_root_missing():
    from morie.entheo.data import load_dmt_imaging
    os.environ["MORIE_DMT_IMAGING_ROOT"] = "/nope/this/does/not/exist"
    try:
        res = load_dmt_imaging()
    finally:
        del os.environ["MORIE_DMT_IMAGING_ROOT"]
    assert len(res["records"]) >= 1
    # Warnings should mention the missing root.
    assert any("DMT_Imaging root" in w for w in res.warnings)


def test_load_subject_id_int_normalises():
    from morie.entheo.data import load_dmt_imaging
    os.environ["MORIE_DMT_IMAGING_ROOT"] = "/nope/this/does/not/exist"
    try:
        res = load_dmt_imaging(subject_id=3)
    finally:
        del os.environ["MORIE_DMT_IMAGING_ROOT"]
    assert res["records"][0]["subject_id"] == "03"
