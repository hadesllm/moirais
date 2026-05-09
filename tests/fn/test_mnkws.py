"""Tests for moirais.fn.mnkws -- Minkowski metric."""

import numpy as np
import pytest

from moirais.fn.mnkws import mnkws


def test_returns_dict():
    r = mnkws(np.zeros(4), np.ones(4))
    assert isinstance(r, dict)
    for k in ("interval_squared", "interval_type", "metric", "separation"):
        assert k in r


def test_lightlike():
    a = np.array([0, 0, 0, 0])
    b = np.array([1, 1, 0, 0])
    r = mnkws(a, b)
    assert r["interval_squared"] == pytest.approx(0.0, abs=1e-12)
    assert r["interval_type"] == "lightlike"


def test_timelike():
    a = np.array([0, 0, 0, 0])
    b = np.array([5, 1, 0, 0])
    r = mnkws(a, b)
    assert r["interval_squared"] > 0
    assert r["interval_type"] == "timelike"


def test_spacelike():
    a = np.array([0, 0, 0, 0])
    b = np.array([0.1, 5, 0, 0])
    r = mnkws(a, b)
    assert r["interval_squared"] < 0
    assert r["interval_type"] == "spacelike"


def test_mostly_plus_signature():
    a = np.array([0, 0, 0, 0])
    b = np.array([5, 1, 0, 0])
    r = mnkws(a, b, signature="mostly_plus")
    assert r["interval_squared"] < 0
    assert r["interval_type"] == "timelike"


def test_wrong_shape():
    with pytest.raises(ValueError):
        mnkws(np.zeros(3), np.zeros(4))
