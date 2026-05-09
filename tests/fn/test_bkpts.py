"""Tests for moirais.fn.bkpts — Bai-Perron structural break detection."""

import numpy as np
import pytest

from moirais.fn.bkpts import bai_perron, bkpts


def test_returns_descriptive_result():
    """Return type has DescriptiveResult interface."""
    y = np.concatenate([np.ones(30), 3 * np.ones(30)])
    r = bai_perron(y, max_breaks=2)
    assert hasattr(r, "value")
    assert hasattr(r, "extra")
    assert "break_dates" in r.extra


def test_obvious_break_detected():
    """A large mean shift should be detected."""
    rng = np.random.default_rng(0)
    # Segment 1: mean 0, Segment 2: mean 10.
    y = np.concatenate([rng.standard_normal(30), 10.0 + rng.standard_normal(30) * 0.5])
    r = bai_perron(y, max_breaks=2)
    assert r.extra["n_breaks"] >= 1
    if r.extra["n_breaks"] >= 1:
        cp = int(r.extra["break_dates"][0])
        assert 15 <= cp <= 45, f"Break should be near 30, got {cp}"


def test_no_break_on_iid():
    """i.i.d. Gaussian — should detect 0 breaks at default settings."""
    rng = np.random.default_rng(42)
    y = rng.standard_normal(100)
    r = bai_perron(y, max_breaks=3)
    assert r.extra["n_breaks"] == 0


def test_segments_cover_series():
    """Segments must cover [0, n) exactly."""
    y = np.concatenate([np.zeros(20), np.ones(20), 2 * np.ones(20)])
    r = bai_perron(y, max_breaks=3)
    segs = r.extra["segments"]
    n = len(y)
    assert segs[0][0] == 0
    assert segs[-1][1] == n
    for i in range(len(segs) - 1):
        assert segs[i][1] == segs[i + 1][0]


def test_n_breaks_le_max_breaks():
    y = np.concatenate([np.zeros(20), np.ones(20), 2 * np.ones(20)])
    r = bai_perron(y, max_breaks=2)
    assert r.extra["n_breaks"] <= 2


def test_coef_by_seg_list():
    """coef_by_seg should have one entry per segment."""
    y = np.concatenate([np.zeros(30), np.ones(30)])
    r = bai_perron(y, max_breaks=2)
    n_segs = r.extra["n_breaks"] + 1
    assert len(r.extra["coef_by_seg"]) == n_segs


def test_too_few_data_points():
    """When data is too short for any break, returns 0 breaks."""
    y = np.ones(10)
    r = bai_perron(y, max_breaks=5, min_size=8)
    assert r.extra["n_breaks"] == 0


def test_alias():
    assert bkpts is bai_perron
