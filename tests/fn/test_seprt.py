"""Tests for morie.fn.seprt — Separation detection."""

import numpy as np
import pytest

from morie.fn.seprt import detect_separation


def test_complete_separation_detected():
    X = np.array([[1], [2], [3], [4], [5], [6]])
    y = np.array([0, 0, 0, 1, 1, 1])
    res = detect_separation(y, X)
    assert res.value is True
    assert res.extra["type"] == "complete"


def test_no_separation():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 1))
    y = (rng.uniform(size=n) < 0.5).astype(float)
    res = detect_separation(y, X)
    assert res.extra["type"] == "none"


def test_separation_reports_variable():
    X = np.array([[1, 0], [2, 0], [3, 0], [4, 1], [5, 1], [6, 1]])
    y = np.array([0, 0, 0, 1, 1, 1])
    res = detect_separation(y, X)
    assert len(res.extra["separating_variables"]) > 0


def test_n_events():
    X = np.ones((10, 1))
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    res = detect_separation(y, X)
    assert res.extra["n_events"] == 5
